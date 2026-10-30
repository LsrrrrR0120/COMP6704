"""Shampoo with Adam grafting (Gupta et al., 2018; Anil et al., 2020).

For a matrix gradient G (m x n) Shampoo keeps Kronecker-factored statistics
L ~ E[G G^T], R ~ E[G^T G] and preconditions as L^{-1/4} G R^{-1/4}. Without
accumulation (beta -> 0) this direction equals U V^T, i.e. Muon's update, which
is why the two methods are compared side by side.

Simplifications kept on purpose for a single-GPU study:
  * statistics are EMAs (shampoo_beta) and inverse roots are recomputed by
    eigendecomposition every `precond_freq` steps;
  * a side larger than `max_precond_dim` is left unpreconditioned and the
    exponent of the remaining side becomes -1/2;
  * the step size/norm is grafted from AdamW (layer-wise), so the lr can be
    shared with AdamW.
"""
import math

import torch
from torch.optim import Optimizer

from .adamw_aux import adamw_update


@torch.no_grad()
def matrix_inverse_root(A, exponent, eps):
    evals, evecs = torch.linalg.eigh(A)
    evals = evals.clamp_min(0) + eps * evals.abs().max().clamp_min(1e-30)
    return (evecs * evals.pow(exponent)) @ evecs.mT


class Shampoo(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        shampoo_beta=0.99,
        eps=1e-8,
        matrix_eps=1e-6,
        precond_freq=20,
        start_precond_step=1,
        max_precond_dim=2048,
        weight_decay=0.0,
    ):
        defaults = dict(
            lr=lr, betas=betas, shampoo_beta=shampoo_beta, eps=eps, matrix_eps=matrix_eps,
            precond_freq=precond_freq, start_precond_step=start_precond_step,
            max_precond_dim=max_precond_dim, weight_decay=weight_decay, use_shampoo=True,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if group["use_shampoo"]:
                    self._shampoo_update(p, group)
                else:
                    adamw_update(p, self.state[p], group["lr"], group["betas"],
                                 group["eps"], group["weight_decay"])
        return loss

    def _shampoo_update(self, p, group):
        grad = p.grad
        state = self.state[p]
        m_dim, n_dim = grad.size(0), grad[0].numel()
        if "step" not in state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)
            for side, dim in (("L", m_dim), ("R", n_dim)):
                use = dim <= group["max_precond_dim"]
                state[side] = torch.zeros(dim, dim, device=p.device) if use else None
                state[side + "_root"] = torch.eye(dim, device=p.device) if use else None
        state["step"] += 1
        t = state["step"]
        beta1, beta2 = group["betas"]

        m, v = state["exp_avg"], state["exp_avg_sq"]
        m.lerp_(grad, 1 - beta1)
        v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        G = grad.reshape(m_dim, n_dim).float()
        sb = group["shampoo_beta"]
        if state["L"] is not None:
            state["L"].mul_(sb).add_(G @ G.T, alpha=1 - sb)
        if state["R"] is not None:
            state["R"].mul_(sb).add_(G.T @ G, alpha=1 - sb)

        sides = [s for s in ("L", "R") if state[s] is not None]
        if sides and t >= group["start_precond_step"] and (t - group["start_precond_step"]) % group["precond_freq"] == 0:
            exponent = -1.0 / (2 * len(sides))
            for s in sides:
                try:
                    state[s + "_root"] = matrix_inverse_root(state[s], exponent, group["matrix_eps"])
                except RuntimeError:  # eigh failure: keep the previous root
                    pass

        bc1, bc2 = 1 - beta1 ** t, 1 - beta2 ** t
        M = (m / bc1).reshape(m_dim, n_dim).float()
        D = M
        if state["L"] is not None:
            D = state["L_root"] @ D
        if state["R"] is not None:
            D = D @ state["R_root"]

        # Layer-wise grafting: Shampoo's direction with AdamW's step norm.
        adam_dir = (m / bc1) / ((v / bc2).sqrt() + group["eps"])
        D = D * (adam_dir.norm() / (D.norm() + 1e-16))

        lr = group["lr"]
        if group["weight_decay"]:
            p.mul_(1 - lr * group["weight_decay"])
        p.add_(D.view_as(p).to(p.dtype), alpha=-lr)
