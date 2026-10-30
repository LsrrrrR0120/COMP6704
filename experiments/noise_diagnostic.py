"""Gradient-noise diagnostic at a fixed parameter point: why does alpha = 0 win at every batch size?

The noisy-gradient model of experiments/spectral_noise_theory.py assumes i.i.d. isotropic
noise and predicts that the best spectral power alpha grows when the batch shrinks. The
alpha x batch-size sweeps found alpha = 0 (polar factor) best everywhere, and the online
SNR estimator of SpectralMuon reported SNRs of 1e-2..1e-4.5. This script freezes the
parameters after a short Muon run and measures, for every hidden matrix:

1. SNR at a fixed point. K micro-batch gradients g_i give the mean G and the per-sample
   noise energy s2; snr(b) = ||G||^2 / (s2 * b0 / b). Unlike the online estimate this
   excludes the drift of the true gradient along the trajectory.
2. Noise structure. Energy fraction of the noise in the top-r (r = 10%) left / right /
   two-sided singular subspaces of G, divided by its value for isotropic noise, and the
   stable rank ||N||_F^2 / ||N||_2^2 of the noise relative to an isotropic Gaussian matrix.
3. One-step test of the theory. For batch size b, X = mean of b / b0 micro-batch gradients
   (real noise) or X = G + isotropic noise of the same Frobenius norm. For each alpha the
   hidden matrices take the step -eta * U S^alpha V^T (Muon scaling) with the best eta on a
   held-out loss; the loss decrease is averaged over disjoint samples.

Usage:  python -m experiments.noise_diagnostic --task gpt   [--out figures/noise]
        python -m experiments.noise_diagnostic --task cifar
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
import torch.nn.functional as F

from common.data import CharDataset, augment_cifar, load_cifar10, normalize_cifar
from common.schedule import lr_factor, set_lr
from common.utils import set_seed
from models import GPT, build_resnet
from optimizers import build_optimizer
from optimizers.muon import muon_scale
from optimizers.spectral_power import spectral_power

ALPHAS = [-0.5, -0.25, 0.0, 0.25, 0.5, 1.0]
ETAS = np.logspace(-4, 0.5, 19)


class Task:
    """Model, micro-batch sampler and held-out loss for one of the two benchmarks."""

    def __init__(self, name, device):
        self.name, self.device = name, device
        if name == "gpt":
            self.ds = CharDataset("data/enwik8", device)
            self.model = GPT(self.ds.vocab_size, 256, 6, 6, 384).to(device)
            self.aux, self.b0, self.train_bs = GPT.AUX_PREFIXES, 8, 64
            self.opt_cfg = dict(name="muon", lr=0.02, aux_lr=3e-3, weight_decay=0.1)
            g = torch.Generator(device=device).manual_seed(1234)
            self.held_out = [self.ds.get_batch("train", 32, 256, generator=g) for _ in range(4)]
        else:
            data = load_cifar10("data", device)
            self.x, self.y = data["train"]
            self.model = build_resnet("resnet18").to(device).to(memory_format=torch.channels_last)
            self.aux, self.b0, self.train_bs = self.model.AUX_PREFIXES, 16, 256
            self.opt_cfg = dict(name="muon", lr=0.05, aux_lr=1e-3, weight_decay=0.05)
            g = torch.Generator(device=device).manual_seed(1234)
            idx = torch.randperm(self.x.size(0), device=device, generator=g)[:2048]
            self.held_out = [(normalize_cifar(self.x[i]), self.y[i]) for i in idx.split(1024)]

    def batch(self, n):
        if self.name == "gpt":
            return self.ds.get_batch("train", n, 256)
        i = torch.randint(self.x.size(0), (n,), device=self.device)
        return augment_cifar(normalize_cifar(self.x[i])), self.y[i]

    def loss(self, xb, yb):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if self.name == "gpt":
                return self.model(xb, yb)
            logits = self.model(xb.contiguous(memory_format=torch.channels_last))
        return F.cross_entropy(logits.float(), yb)

    @torch.no_grad()
    def held_out_loss(self):
        was_training = self.model.training
        self.model.eval()
        out = sum(self.loss(x, y).item() for x, y in self.held_out) / len(self.held_out)
        self.model.train(was_training)
        return out

    def hidden(self):
        return [(n, p) for n, p in self.model.named_parameters()
                if p.ndim >= 2 and not n.startswith(tuple(self.aux))]


def pretrain(task, steps):
    """Muon with warmup + constant lr, so the diagnostic point is mid-training."""
    c = task.opt_cfg
    opt = build_optimizer(c["name"], task.model, c["lr"], weight_decay=c["weight_decay"],
                          aux_prefixes=task.aux, aux_lr=c["aux_lr"])
    task.model.train()
    for step in range(steps):
        set_lr(opt, lr_factor(step, steps, steps // 10, "constant"))
        opt.zero_grad(set_to_none=True)
        task.loss(*task.batch(task.train_bs)).backward()
        opt.step()
        if step % 200 == 0 or step == steps - 1:
            print(f"  pretrain step {step:5d}  held-out loss {task.held_out_loss():.4f}")


def sample_grads(task, K):
    """K micro-batch gradients of every hidden matrix, as tall 2-D bf16 tensors."""
    hidden = task.hidden()
    grads = {n: torch.empty(K, *tall_shape(p), dtype=torch.bfloat16, device=task.device) for n, p in hidden}
    for k in range(K):
        task.model.zero_grad(set_to_none=True)
        task.loss(*task.batch(task.b0)).backward()
        for n, p in hidden:
            grads[n][k] = tall(p.grad)
    task.model.zero_grad(set_to_none=True)
    return grads


def tall_shape(p):
    r, c = p.size(0), p[0].numel()
    return (r, c) if r >= c else (c, r)


def tall(t):
    t = t.reshape(t.size(0), -1)
    return t if t.size(0) >= t.size(1) else t.mT


def untall(t, p):
    return (t if p.size(0) >= p[0].numel() else t.mT).reshape_as(p)


def noise_stats(g):
    """SNR at micro-batch size and noise-structure ratios for one matrix (g: K x m x n)."""
    K, m, n = g.shape
    g = g.float()
    G = g.mean(0)
    N = g - G
    s2 = N.pow(2).sum((1, 2)).sum().item() / (K - 1)
    signal = max(G.pow(2).sum().item() - s2 / K, 1e-30)
    r = max(1, math.ceil(0.1 * n))
    U, _, Vh = torch.linalg.svd(G, full_matrices=False)
    Ur, Vr = U[:, :r], Vh[:r].mT
    e = N.pow(2).sum((1, 2))
    fl = ((Ur.mT @ N).pow(2).sum((1, 2)) / e).mean().item() / (r / m)
    fr = ((N @ Vr).pow(2).sum((1, 2)) / e).mean().item() / (r / n)
    flr = ((Ur.mT @ N @ Vr).pow(2).sum((1, 2)) / e).mean().item() / (r * r / (m * n))
    srank = (e[:8] / torch.linalg.matrix_norm(N[:8], 2).pow(2)).mean().item()
    iso = torch.randn(8, m, n, device=g.device)
    srank_iso = (iso.pow(2).sum((1, 2)) / torch.linalg.matrix_norm(iso, 2).pow(2)).mean().item()
    return dict(shape=[m, n], log10_snr_b0=math.log10(signal / s2), left=fl, right=fr, both=flr,
                stable_rank_ratio=srank / srank_iso)


@torch.no_grad()
def one_step_test(task, grads, batch_sizes, samples, gen):
    hidden = task.hidden()
    theta0 = {n: p.detach().clone() for n, p in hidden}
    Gbar = {n: g.float().mean(0) for n, g in grads.items()}
    L0 = task.held_out_loss()

    def best_decrease(X):
        out = []
        for a in ALPHAS:
            D = {n: spectral_power(X[n], a) * muon_scale(p.size(0), p[0].numel(), "spectral") for n, p in hidden}
            dec = []
            for eta in ETAS:
                for n, p in hidden:
                    p.copy_(theta0[n] - eta * untall(D[n], p).to(p.dtype))
                dec.append(L0 - task.held_out_loss())
            k = int(np.argmax(dec))
            if k in (0, len(ETAS) - 1) and dec[k] > 0:
                print(f"    warning: alpha={a:+.2f} best eta {ETAS[k]:.1e} is on the grid boundary")
            out.append(max(dec[k], 0.0))
        for n, p in hidden:
            p.copy_(theta0[n])
        return out

    K = next(iter(grads.values())).size(0)
    res = {"full": best_decrease(Gbar)}
    for b in batch_sizes:
        mult = b // task.b0
        real, iso = [], []
        for s in range(min(samples, K // mult)):
            X = {n: g[s * mult:(s + 1) * mult].float().mean(0) for n, g in grads.items()}
            Z = {n: torch.randn(x.shape, device=x.device, generator=gen) for n, x in X.items()}
            Xiso = {n: Gbar[n] + (X[n] - Gbar[n]).norm() * Z[n] / Z[n].norm() for n in X}
            real.append(best_decrease(X))
            iso.append(best_decrease(Xiso))
        res[b] = dict(real=np.mean(real, 0).tolist(), iso=np.mean(iso, 0).tolist())
        print(f"  b={b:4d}  best alpha real {ALPHAS[int(np.argmax(res[b]['real']))]:+.2f}"
              f"  iso {ALPHAS[int(np.argmax(res[b]['iso']))]:+.2f}   "
              f"decrease real {np.round(res[b]['real'], 4).tolist()}  iso {np.round(res[b]['iso'], 4).tolist()}")
    return L0, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["gpt", "cifar"], required=True)
    ap.add_argument("--pretrain-steps", type=int, default=None)
    ap.add_argument("--K", type=int, default=64, help="micro-batch gradients at the fixed point")
    ap.add_argument("--samples", type=int, default=4, help="disjoint samples per batch size in the one-step test")
    ap.add_argument("--out", default="figures/noise")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_seed(0)
    device = torch.device("cuda")
    task = Task(args.task, device)
    steps = args.pretrain_steps or (1000 if args.task == "gpt" else 400)
    print(f"[{args.task}] pretraining {steps} Muon steps")
    pretrain(task, steps)

    grads = sample_grads(task, args.K)
    stats = {n: noise_stats(g) for n, g in grads.items()}
    batch_sizes = [task.b0 * m for m in (1, 2, 4, 8, 16)]
    snr = np.array([s["log10_snr_b0"] for s in stats.values()])
    print(f"  log10 SNR at b0={task.b0}: mean {snr.mean():.2f}  range [{snr.min():.2f}, {snr.max():.2f}]"
          f"  -> at train batch {task.train_bs}: {snr.mean() + math.log10(task.train_bs / task.b0):.2f}")
    for key in ("left", "right", "both", "stable_rank_ratio"):
        v = np.array([s[key] for s in stats.values()])
        print(f"  {key:18s} mean {v.mean():.2f}  range [{v.min():.2f}, {v.max():.2f}]")

    gen = torch.Generator(device=device).manual_seed(0)
    L0, res = one_step_test(task, grads, batch_sizes, args.samples, gen)
    result = dict(task=args.task, pretrain_steps=steps, K=args.K, b0=task.b0, held_out_loss=L0,
                  alphas=ALPHAS, layers=stats, one_step={str(k): v for k, v in res.items()})
    with open(os.path.join(args.out, f"noise_{args.task}.json"), "w") as f:
        json.dump(result, f, indent=1)
    plot(result, os.path.join(args.out, f"noise_{args.task}"))
    print(f"written to {args.out}")


def plot(r, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    names = list(r["layers"])
    b0 = r["b0"]
    snr = np.array([r["layers"][n]["log10_snr_b0"] for n in names])
    bs = np.array([b0 * m for m in (1, 2, 4, 8, 16, 32, 64)])
    for n, s in zip(names, snr):
        axes[0].plot(bs, s + np.log10(bs / b0), color="0.75", lw=0.8)
    axes[0].plot(bs, snr.mean() + np.log10(bs / b0), "k-o", ms=4, label="mean over hidden matrices")
    axes[0].axhline(0, ls=":", color="r")
    axes[0].set(xscale="log", xlabel="batch size b", ylabel=r"$\log_{10}$ SNR $=\|G\|_F^2/E\|N_b\|_F^2$",
                title="SNR at a fixed point (per hidden matrix)")
    axes[0].legend(fontsize=7)

    keys = ["left", "right", "both", "stable_rank_ratio"]
    labels = ["top-10% left", "top-10% right", "top-10% both", "stable rank"]
    vals = np.array([[r["layers"][n][k] for k in keys] for n in names])
    axes[1].boxplot([vals[:, i] for i in range(len(keys))], tick_labels=labels)
    axes[1].axhline(1, ls=":", color="r")
    axes[1].set(yscale="log", ylabel="ratio to isotropic noise", title="Noise structure relative to isotropic")

    one = r["one_step"]
    full = max(one["full"])
    keys_b = [k for k in one if k != "full"]
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(keys_b)))
    for c, k in zip(colors, keys_b):
        axes[2].plot(r["alphas"], np.array(one[k]["real"]) / full, "o-", color=c, ms=4, label=f"b={k} real")
        axes[2].plot(r["alphas"], np.array(one[k]["iso"]) / full, "x--", color=c, ms=4, label=f"b={k} isotropic")
    axes[2].plot(r["alphas"], np.array(one["full"]) / full, "k-", lw=2, label="mean gradient")
    axes[2].set(xlabel=r"$\alpha$", ylabel="best one-step loss decrease (normalized)",
                title="One-step test: real vs isotropic noise")
    axes[2].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
