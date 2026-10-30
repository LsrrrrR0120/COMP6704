"""Muon: momentum + orthogonalization, i.e. steepest descent under the spectral norm.

Reference: K. Jordan et al., "Muon: An optimizer for hidden layers in neural
networks", 2024; J. Bernstein & L. Newhouse, "Old optimizer, new norm", 2024.

Parameter groups with use_muon=True must hold matrices (conv kernels are
flattened to out_channels x (in_channels * k * k)). All other groups are
updated with AdamW, as recommended for embeddings, output heads and vectors.
"""
import torch
from torch.optim import Optimizer

from .adamw_aux import adamw_update
from .orthogonalize import orthogonalize


def muon_scale(rows, cols, mode):
    if mode == "spectral":
        # Keeps the update's RMS->RMS operator norm ~1 for tall matrices (Jordan et al.).
        return max(1.0, rows / cols) ** 0.5
    if mode == "rms":
        # Matches the per-entry RMS of a typical AdamW update (~0.2), so the
        # same lr can be shared with AdamW (Liu et al., "Muon is scalable", 2025).
        return 0.2 * max(rows, cols) ** 0.5
    raise ValueError(f"unknown scale mode: {mode}")


class Muon(Optimizer):
    def __init__(
        self,
        params,
        lr=0.02,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        orth_method="ns",
        scale_mode="spectral",
        weight_decay=0.0,
        adam_betas=(0.9, 0.95),
        adam_eps=1e-8,
    ):
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
            orth_method=orth_method, scale_mode=scale_mode, weight_decay=weight_decay,
            adam_betas=adam_betas, adam_eps=adam_eps, use_muon=True,
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
                if group["use_muon"]:
                    self._muon_update(p, group)
                else:
                    adamw_update(p, self.state[p], group["lr"], group["adam_betas"],
                                 group["adam_eps"], group["weight_decay"])
        return loss

    def _muon_update(self, p, group):
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(p)
        beta = group["momentum"]
        buf = state["momentum_buffer"].mul_(beta).add_(p.grad)
        update = p.grad.add(buf, alpha=beta) if group["nesterov"] else buf
        G = update.reshape(update.size(0), -1)
        O = orthogonalize(G, group["orth_method"], group["ns_steps"])
        O = O * muon_scale(G.size(0), G.size(1), group["scale_mode"])
        lr = group["lr"]
        if group["weight_decay"]:
            p.mul_(1 - lr * group["weight_decay"])
        p.add_(O.view_as(p), alpha=-lr)
