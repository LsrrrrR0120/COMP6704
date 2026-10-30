"""Spectral-power Muon: update U S^alpha V^T for the momentum X = U S V^T.

alpha = 0 is the exact polar factor (Muon's target), alpha = 1 is the momentum itself
(normalized SGD with momentum). For alpha in (0, 1) the update is the steepest-descent
direction under the Schatten-q norm with q = 1 + 1/alpha (cf. Freon, arXiv 2605.11181;
Muon^p, arXiv 2606.13867). Small singular directions get weight sigma^alpha instead
of 1; the report studies how the best alpha depends on mini-batch gradient noise.

Computed exactly from the eigendecomposition of the Gram matrix of the smaller side
(X^T X = V S^2 V^T for tall X), so U S^alpha V^T = X V diag(sigma^(alpha-1)) V^T.
The update is rescaled to the Frobenius norm of the polar factor (sqrt(rank)), so
alpha changes the shape of the update but not its size and learning rates stay comparable.

Noise-adaptive mode (alpha="adaptive", our proposal). Each layer estimates the
signal-to-noise ratio of the matrix X it maps, from two scalar EMAs:
    E||g||^2 = ||G||^2 + s2,     E||m||^2 = ||G||^2 + s2 (1 - beta) / (1 + beta),
where g is the stochastic gradient, m = (1 - beta) * buf the EMA momentum, G the true
gradient and s2 the per-step noise energy. Solving gives ||G||^2 and s2; the Nesterov
combination X = (1 + beta) g + beta^2 buf_prev amplifies the signal by c_s and the noise
by sqrt(c_n), so snr_X = c_s^2 ||G||^2 / (c_n s2). The spectral power then follows the
noisy-gradient model of experiments/spectral_noise_theory.py, where the best alpha falls
roughly linearly in log(SNR):
    alpha = clip(1 - slope * log10(snr_X / snr0), 0, 1).
Everything stays on the GPU (no host synchronization).
"""
import math

import torch

from .adamw_aux import adamw_update
from .muon import Muon, muon_scale


@torch.no_grad()
def spectral_power(X, alpha, sigma_floor=1e-3):
    """U diag(w) V^T with w = sigma^alpha for a tall X, rescaled to ||.||_F = sqrt(rank).
    `alpha` may be a float or a 0-dim tensor."""
    X = X / (X.norm() + 1e-12)
    evals, V = torch.linalg.eigh(X.mT @ X)
    return _apply_power(X, evals, V, alpha, sigma_floor)


def _apply_power(X, evals, V, alpha, sigma_floor):
    """Given the eigendecomposition V diag(evals) V^T of X^T X (X Frobenius-normalized)."""
    sigma = evals.clamp_min(0).sqrt()
    keep = sigma > sigma_floor * sigma.max()
    w = torch.where(keep, sigma.clamp_min(1e-30).pow(alpha), torch.zeros_like(sigma))
    coef = torch.where(keep, w / sigma.clamp_min(1e-30), torch.zeros_like(sigma))
    D = (X @ V) * coef @ V.mT
    return D * (keep.sum().sqrt() / (w.norm() + 1e-30))


class SpectralMuon(Muon):
    def __init__(self, params, lr=0.02, alpha=0.5, sigma_floor=1e-3,
                 snr_beta=0.99, snr0=1.0, slope=0.35, warmup_alpha=0.0, **kwargs):
        super().__init__(params, lr=lr, **kwargs)
        for g in self.param_groups:
            g.setdefault("alpha", alpha)
            g.setdefault("sigma_floor", sigma_floor)
            g.setdefault("snr_beta", snr_beta)
            g.setdefault("snr0", snr0)
            g.setdefault("slope", slope)
            g.setdefault("warmup_alpha", warmup_alpha)

    @torch.no_grad()
    def step(self, closure=None):
        # Same as Muon.step, except that the eigendecompositions of all matrices whose
        # Gram matrices have the same size run as one batched eigh (~3x faster on GPU).
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        buckets = {}
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if not group["use_muon"]:
                    adamw_update(p, self.state[p], group["lr"], group["adam_betas"],
                                 group["adam_eps"], group["weight_decay"])
                    continue
                X, alpha = self._prepare(p, group)
                buckets.setdefault(X.size(1), []).append((p, group, X, alpha))
        for items in buckets.values():
            evals, V = torch.linalg.eigh(torch.stack([X.mT @ X for _, _, X, _ in items]))
            for (p, group, X, alpha), e, v in zip(items, evals, V):
                self._apply(p, group, _apply_power(X, e, v, alpha, group["sigma_floor"]))
        return loss

    def _prepare(self, p, group):
        """Momentum step; returns the tall, Frobenius-normalized matrix to transform and its alpha."""
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(p)
            state["step"] = 0
        beta = group["momentum"]
        buf = state["momentum_buffer"].mul_(beta).add_(p.grad)
        update = p.grad.add(buf, alpha=beta) if group["nesterov"] else buf
        state["step"] += 1

        alpha = group["alpha"]
        if alpha == "adaptive":
            alpha = self._adaptive_alpha(p, buf, state, group)
        X = update.reshape(update.size(0), -1).float()
        if X.size(0) < X.size(1):
            X = X.mT
        return X / (X.norm() + 1e-12), alpha

    def _apply(self, p, group, D):
        rows, cols = p.size(0), p[0].numel()
        if rows < cols:
            D = D.mT
        D = D * muon_scale(rows, cols, group["scale_mode"])
        lr = group["lr"]
        if group["weight_decay"]:
            p.mul_(1 - lr * group["weight_decay"])
        p.add_(D.reshape_as(p).to(p.dtype), alpha=-lr)

    def _adaptive_alpha(self, p, buf, state, group):
        beta, rho = group["momentum"], group["snr_beta"]
        g2 = p.grad.float().pow(2).sum()
        m2 = ((1 - beta) * buf.float()).pow(2).sum()
        if "ema_g2" not in state:
            state["ema_g2"], state["ema_m2"] = torch.zeros_like(g2), torch.zeros_like(m2)
        state["ema_g2"].lerp_(g2, 1 - rho)
        state["ema_m2"].lerp_(m2, 1 - rho)
        bc = 1 - rho ** state["step"]
        eg2, em2 = state["ema_g2"] / bc, state["ema_m2"] / bc
        noise = ((eg2 - em2) * (1 + beta) / (2 * beta)).clamp_min(1e-30 + 1e-6 * eg2)
        signal = (eg2 - noise).clamp_min(1e-6 * eg2)
        c_s = (1 + beta) + beta ** 2 / (1 - beta)
        c_n = (1 + beta) ** 2 + beta ** 4 / (1 - beta ** 2)
        snr = c_s ** 2 * signal / (c_n * noise)
        alpha = (1 - group["slope"] * torch.log10(snr / group["snr0"])).clamp(0.0, 1.0)
        # The momentum EMA needs ~1/(1-beta) steps before the two-moment estimate is valid.
        if state["step"] < 2 / (1 - beta) + 1 / (1 - rho) / 2:
            alpha = torch.full_like(alpha, group["warmup_alpha"])
        state["snr"], state["alpha"] = snr, alpha
        return alpha

    def adaptive_report(self):
        """Mean and range of the per-layer SNR / alpha (syncs; call only when logging)."""
        snr = [s["snr"] for s in self.state.values() if "snr" in s]
        alpha = [s["alpha"] for s in self.state.values() if "alpha" in s]
        if not snr:
            return {}
        snr, alpha = torch.stack(snr).log10(), torch.stack(alpha)
        return dict(log10_snr_mean=snr.mean().item(), log10_snr_min=snr.min().item(),
                    log10_snr_max=snr.max().item(), alpha_mean=alpha.mean().item(),
                    alpha_min=alpha.min().item(), alpha_max=alpha.max().item())
