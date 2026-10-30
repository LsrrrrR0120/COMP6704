"""Does a half-batch SNR estimator recover the fixed-point SNR that the two-moment EMA misses?

The noise-adaptive SpectralMuon estimated each layer's SNR from EMAs of ||g||^2 and of the
momentum norm across steps; experiments/noise_diagnostic.py showed that this is 2-5 orders
of magnitude too low, because the true gradient drifts between steps. Here every training
batch of size b is split into two halves with gradients g1, g2 at the same parameters:
    g = (g1 + g2) / 2,   E||g1 - g2||^2 / 4 = E||g - G||^2 = noise energy at batch b,
so N = ||g1 - g2||^2 / 4 and S = ||g||^2 - N are unbiased for the noise and signal energy
of the batch-b gradient, with no drift term. Per layer, snr_b = EMA(S) / EMA(N).

The script reruns the Muon pretraining of noise_diagnostic.py (same seed, batch, lr),
logs the half-batch and the two-moment estimates along the way, and at the end compares
both with the fixed-point SNR from K micro-batch gradients at the final parameters.

Usage:  python -m experiments.snr_estimator_check --task gpt   [--out figures/snr_check]
        python -m experiments.snr_estimator_check --task cifar
"""
import argparse
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common.schedule import lr_factor, set_lr
from common.utils import set_seed
from experiments.noise_diagnostic import Task, noise_stats, sample_grads
from optimizers import build_optimizer


class SNRTracker:
    """Per-layer half-batch and two-moment SNR estimates of the batch-b gradient."""

    def __init__(self, params, rho=0.95, beta=0.95, rho_old=0.99):
        self.params, self.rho, self.beta, self.rho_old = params, rho, beta, rho_old
        z = lambda: torch.zeros(len(params), device=params[0].device)
        self.S, self.N, self.g2, self.m2 = z(), z(), z(), z()
        self.buf = [torch.zeros_like(p, dtype=torch.float32) for p in params]
        self.step = 0

    @torch.no_grad()
    def update(self, half):
        """`half`: gradients after the first half-batch backward (each half scaled by 1/2)."""
        self.step += 1
        S, N, g2, m2 = [], [], [], []
        for p, h, b in zip(self.params, half, self.buf):
            g = p.grad.float()
            d = 4 * h - 2 * g                     # g1 - g2, with g1 = 2h, g2 = 2(g - h)
            n = d.pow(2).sum() / 4
            gg = g.pow(2).sum()
            S.append(gg - n)
            N.append(n)
            b.mul_(self.beta).add_(g)
            g2.append(gg)
            m2.append(((1 - self.beta) * b).pow(2).sum())
        self.S.lerp_(torch.stack(S), 1 - self.rho)
        self.N.lerp_(torch.stack(N), 1 - self.rho)
        self.g2.lerp_(torch.stack(g2), 1 - self.rho_old)
        self.m2.lerp_(torch.stack(m2), 1 - self.rho_old)

    def half_batch(self):
        bc = 1 - self.rho ** self.step
        S, N = self.S / bc, self.N / bc
        return torch.log10(S.clamp_min(1e-6 * N) / N)

    def two_moment(self):
        """The estimator of SpectralMuon._adaptive_alpha, for the batch-b gradient (not X)."""
        bc = 1 - self.rho_old ** self.step
        eg2, em2 = self.g2 / bc, self.m2 / bc
        beta = self.beta
        noise = ((eg2 - em2) * (1 + beta) / (2 * beta)).clamp_min(1e-30 + 1e-6 * eg2)
        signal = (eg2 - noise).clamp_min(1e-6 * eg2)
        return torch.log10(signal / noise)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["gpt", "cifar"], required=True)
    ap.add_argument("--pretrain-steps", type=int, default=None)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--rho", type=float, default=0.95, help="EMA factor of the half-batch estimator")
    ap.add_argument("--out", default="figures/snr_check")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_seed(0)
    device = torch.device("cuda")
    task = Task(args.task, device)
    steps = args.pretrain_steps or (1000 if args.task == "gpt" else 400)
    c = task.opt_cfg
    opt = build_optimizer(c["name"], task.model, c["lr"], weight_decay=c["weight_decay"],
                          aux_prefixes=task.aux, aux_lr=c["aux_lr"])
    names, params = zip(*task.hidden())
    tracker = SNRTracker(list(params), rho=args.rho)
    trace = []
    task.model.train()
    for step in range(steps):
        set_lr(opt, lr_factor(step, steps, steps // 10, "constant"))
        opt.zero_grad(set_to_none=True)
        xb, yb = task.batch(task.train_bs)
        h = task.train_bs // 2
        (0.5 * task.loss(xb[:h], yb[:h])).backward()
        half = [p.grad.float().clone() for p in params]
        (0.5 * task.loss(xb[h:], yb[h:])).backward()
        tracker.update(half)
        opt.step()
        if step % 20 == 0 or step == steps - 1:
            hb, tm = tracker.half_batch(), tracker.two_moment()
            trace.append(dict(step=step, half_batch=hb.mean().item(), two_moment=tm.mean().item()))
        if step % 200 == 0 or step == steps - 1:
            print(f"  step {step:5d}  held-out {task.held_out_loss():.4f}  log10 SNR_b  "
                  f"half-batch {trace[-1]['half_batch']:+.2f}  two-moment {trace[-1]['two_moment']:+.2f}")

    hb = tracker.half_batch().cpu().numpy()
    tm = tracker.two_moment().cpu().numpy()
    grads = sample_grads(task, args.K)
    shift = math.log10(task.train_bs / task.b0)
    truth = np.array([noise_stats(grads[n])["log10_snr_b0"] + shift for n in names])
    err_hb, err_tm = hb - truth, tm - truth
    print(f"  fixed-point log10 SNR at b={task.train_bs}: mean {truth.mean():+.2f}")
    print(f"  half-batch : mean {hb.mean():+.2f}  error mean {err_hb.mean():+.2f}  "
          f"|error| median {np.median(np.abs(err_hb)):.2f}  corr {np.corrcoef(hb, truth)[0, 1]:.2f}")
    print(f"  two-moment : mean {tm.mean():+.2f}  error mean {err_tm.mean():+.2f}  "
          f"|error| median {np.median(np.abs(err_tm)):.2f}  corr {np.corrcoef(tm, truth)[0, 1]:.2f}")
    result = dict(task=args.task, steps=steps, train_bs=task.train_bs, rho=args.rho, layers=list(names),
                  truth=truth.tolist(), half_batch=hb.tolist(), two_moment=tm.tolist(), trace=trace)
    path = os.path.join(args.out, f"snr_check_{args.task}")
    with open(path + ".json", "w") as f:
        json.dump(result, f, indent=1)
    plot(result, path)
    print(f"written to {args.out}")


def plot(r, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    t = np.array(r["truth"])
    lo, hi = min(t.min(), min(r["half_batch"])) - 0.3, t.max() + 0.3
    axes[0].plot([lo, hi], [lo, hi], "k:", lw=1)
    axes[0].scatter(t, r["half_batch"], s=14, label="half-batch (new)")
    axes[0].scatter(t, r["two_moment"], s=14, marker="x", label="two-moment EMA (old)")
    axes[0].set(xlabel=r"fixed-point $\log_{10}$ SNR$_b$", ylabel=r"online estimate",
                title=f"{r['task']}: per hidden matrix at step {r['steps']}")
    axes[0].legend(fontsize=8)
    s = [p["step"] for p in r["trace"]]
    axes[1].plot(s, [p["half_batch"] for p in r["trace"]], label="half-batch (new)")
    axes[1].plot(s, [p["two_moment"] for p in r["trace"]], label="two-moment EMA (old)")
    axes[1].axhline(t.mean(), color="k", ls=":", label="fixed point (end)")
    axes[1].set(xlabel="step", ylabel=r"mean $\log_{10}$ SNR$_b$", title="estimate during training")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path + ".png", dpi=150)
    fig.savefig(path + ".pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
