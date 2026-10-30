"""WS-Muon: Muon with a warm-started (cached) inverse square root of the momentum Gram.

For a tall matrix X (m >= n) with polar decomposition X = O H, the polar factor is
O = X (X^T X)^{-1/2}. Muon recomputes an approximation of O from scratch at every
step with 5 quintic Newton-Schulz iterations. The momentum changes slowly, so WS-Muon

  1. caches a symmetric P ~ (X^T X + damping * lambda_max I)^{-1/2}, recomputed by an
     eigendecomposition only every `refresh_every` steps (or when the cache is too stale);
  2. at every step forms Y = X P, whose singular values are already close to 1;
  3. applies `correction_steps` cheap cubic Newton-Schulz steps Y <- Y (3I - Y^T Y)/2.

Why this is accurate (see the report): write X P = O (I + F) with F small. The cubic
step squares the singular-value error (|1 - s| -> ~1.5 |1 - s|^2), and since H is
symmetric the alignment loss of polar(X P) w.r.t. X is only O(||F||^2) -- the first-order
term tr(H skew(F)) vanishes. Without the correction step (correction_steps=0) the
update is a CacheMuon-style reuse of a stale transform (arXiv 2606.16371).

Cost per step for an m x n matrix (n = smaller side): 1 + 2 * correction_steps
products of size m x n x n, versus 5 * (2 m n^2 + n^3) for Muon's 5 quintic steps.
"""
import math

import torch

from .muon import Muon, muon_scale


@torch.no_grad()
def damped_inverse_sqrt(A, damping):
    evals, evecs = torch.linalg.eigh(A)
    evals = evals.clamp_min(0) + damping * evals.max().clamp_min(1e-30)
    return (evecs * evals.rsqrt()) @ evecs.mT


class WSMuon(Muon):
    def __init__(self, params, lr=0.02, refresh_every=10, correction_steps=1, damping=1e-3,
                 refresh_tol=0.5, **kwargs):
        super().__init__(params, lr=lr, **kwargs)
        for g in self.param_groups:
            g.setdefault("refresh_every", refresh_every)
            g.setdefault("correction_steps", correction_steps)
            g.setdefault("damping", damping)
            g.setdefault("refresh_tol", refresh_tol)
        self.stats = dict(steps=0, refreshes=0)  # matrix updates / eigendecompositions

    def _muon_update(self, p, group):
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(p)
            state["since_refresh"] = math.inf
        beta = group["momentum"]
        buf = state["momentum_buffer"].mul_(beta).add_(p.grad)
        update = p.grad.add(buf, alpha=beta) if group["nesterov"] else buf

        G = update.reshape(update.size(0), -1).float()
        rows, cols = G.shape
        O = self.orthogonalize_cached(G.mT if rows < cols else G, state, group)
        O = O.mT if rows < cols else O
        O = O * muon_scale(rows, cols, group["scale_mode"])

        lr = group["lr"]
        if group["weight_decay"]:
            p.mul_(1 - lr * group["weight_decay"])
        p.add_(O.view_as(p).to(p.dtype), alpha=-lr)

    def orthogonalize_cached(self, X, state, group):
        """Approximate polar factor of a tall matrix X (m >= n) using the cached P."""
        X = X / (X.norm() + 1e-7)
        n = X.size(1)
        eye = torch.eye(n, device=X.device, dtype=X.dtype)
        self.stats["steps"] += 1

        if state["since_refresh"] >= group["refresh_every"]:
            self._refresh(X, state, group)
        Y = X @ state["P"]
        B = Y.mT @ Y
        # Rescale so the mean squared singular value of Y is 1 (absorbs global scale drift).
        s2 = B.diagonal().mean().clamp_min(1e-12)
        residual = ((B / s2 - eye).norm() / math.sqrt(n)).item() if group["refresh_tol"] else 0.0
        if group["refresh_tol"] and residual > group["refresh_tol"] and state["since_refresh"] > 0:
            self._refresh(X, state, group)  # cache too stale: recompute and redo the product
            Y = X @ state["P"]
            B = Y.mT @ Y
            s2 = B.diagonal().mean().clamp_min(1e-12)
        Y, B = Y / s2.sqrt(), B / s2

        for k in range(group["correction_steps"]):
            if k > 0:
                B = Y.mT @ Y
            Y = Y @ (1.5 * eye - 0.5 * B)
        state["since_refresh"] += 1
        return Y

    def _refresh(self, X, state, group):
        state["P"] = damped_inverse_sqrt(X.mT @ X, group["damping"])
        state["since_refresh"] = 0
        self.stats["refreshes"] += 1
