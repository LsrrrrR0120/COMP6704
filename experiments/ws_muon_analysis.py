"""Offline analysis of WS-Muon's cached orthogonalization on real momentum sequences.

1. Train a small GPT with Muon on Tiny Shakespeare and record, at every step, the
   Nesterov momentum matrix that Muon orthogonalizes (for a few layers).
2. Replay each sequence through different orthogonalizers and measure
     alignment(X, O) = <X, O> / (||X||_* * ||O||_2)   (1 = exact steepest descent direction)
     orthogonality error ||O^T O / s^2 - I||_F / sqrt(n)
   for exact polar (SVD), Muon's 5 quintic NS steps, and WS-Muon with refresh period k
   and s cubic correction steps (s = 0 is a CacheMuon-style stale reuse).
3. Check the theory: 1 - alignment should scale like r^(2^s)-ish in the cache residual r
   (first order for s = 0, second order or better for s >= 1).

Usage:  python -m experiments.ws_muon_analysis [--steps 400] [--device cpu]
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

from common.data import CharDataset
from common.utils import set_seed
from models import GPT
from optimizers import build_optimizer
from optimizers.orthogonalize import newton_schulz, polar_svd
from optimizers.ws_muon import WSMuon

TRACKED = ["blocks.0.attn.q.weight", "blocks.1.fc.weight", "blocks.1.fc_out.weight"]


def record_momenta(args, device):
    """Train with Muon and return {param name: [nesterov momentum matrix per step]}."""
    set_seed(0)
    ds = CharDataset("data/tinyshakespeare.txt", device)
    model = GPT(ds.vocab_size, block_size=64, n_layer=2, n_head=4, n_embd=128).to(device)
    opt = build_optimizer("muon", model, 0.02, weight_decay=0.0, aux_prefixes=GPT.AUX_PREFIXES, aux_lr=3e-3)
    params = dict(model.named_parameters())
    seqs = {n: [] for n in TRACKED}
    for step in range(args.steps):
        x, y = ds.get_batch("train", 32, 64)
        loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for n in TRACKED:  # same Nesterov combination Muon is about to orthogonalize
            p = params[n]
            buf = opt.state[p]["momentum_buffer"] if p in opt.state else torch.zeros_like(p)
            beta = 0.95
            seqs[n].append((p.grad + beta * (beta * buf + p.grad)).detach().clone())
        opt.step()
        if step % 100 == 0:
            print(f"  step {step} loss {loss.item():.3f}")
    return seqs


def alignment(X, O):
    nuc = torch.linalg.matrix_norm(X, ord="nuc")
    spec = torch.linalg.matrix_norm(O, ord=2)
    return ((X * O).sum() / (nuc * spec)).item()


def ortho_error(O):
    B = O.mT @ O if O.size(0) >= O.size(1) else O @ O.mT
    B = B / B.diagonal().mean()
    return ((B - torch.eye(B.size(0))).norm() / math.sqrt(B.size(0))).item()


def replay_ws(seq, k, s, damping=1e-3):
    """Run WSMuon's real orthogonalization code over a recorded momentum sequence."""
    dummy = torch.nn.Parameter(torch.zeros(1, 1))
    opt = WSMuon([dummy], refresh_every=k, correction_steps=s, damping=damping, refresh_tol=None)
    group, state = opt.param_groups[0], {"since_refresh": math.inf}
    aligns, errs, resid = [], [], []
    for G in seq:
        G = G.float()
        tall = G.size(0) >= G.size(1)
        X = G if tall else G.mT
        Xn = X / X.norm()
        if "P" in state and state["since_refresh"] < k:  # residual of the cache before correction
            Y = Xn @ state["P"]
            B = Y.mT @ Y
            B = B / B.diagonal().mean()
            resid.append(((B - torch.eye(B.size(0))).norm() / math.sqrt(B.size(0))).item())
        else:
            resid.append(0.0)
        O = opt.orthogonalize_cached(X, state, group)
        O = O if tall else O.mT
        aligns.append(alignment(G, O))
        errs.append(ortho_error(O))
    return np.array(aligns), np.array(errs), np.array(resid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="figures/ws_muon")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    device = torch.device(args.device)

    print("[1] recording momentum matrices along a Muon run")
    seqs = {n: [g.cpu() for g in v] for n, v in record_momenta(args, device).items()}
    burn = 20  # skip the first steps where the momentum buffer is still filling

    print("[2] replaying orthogonalizers")
    ks, ss = [1, 2, 5, 10, 20, 50, 100], [0, 1, 2]
    summary = {}
    for name, seq in seqs.items():
        seq = seq[burn:]
        res = {"exact": float(np.mean([alignment(G, polar_svd(G)) for G in seq])),
               "muon_ns5": float(np.mean([alignment(G, newton_schulz(G, 5).float()) for G in seq])),
               "muon_ns5_ortho_err": float(np.mean([ortho_error(newton_schulz(G, 5).float()) for G in seq]))}
        for s in ss:
            for k in ks:
                a, e, _ = replay_ws(seq, k, s)
                res[f"ws_k{k}_s{s}"] = float(a.mean())
                res[f"ws_k{k}_s{s}_ortho_err"] = float(e.mean())
        summary[name] = res
        print(f"  {name} {tuple(seq[0].shape)}: exact {res['exact']:.4f}  NS5 {res['muon_ns5']:.4f}  "
              + "  ".join(f"k{k}/s1 {res[f'ws_k{k}_s1']:.4f}" for k in ks))

    # ---- figure: alignment vs refresh period
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    for name, res in summary.items():
        short = name.replace(".weight", "")
        for s, ls in zip(ss, [":", "-", "--"]):
            axes[0].semilogx(ks, [res[f"ws_k{k}_s{s}"] for k in ks], ls, marker="o", ms=3,
                             label=f"{short}, s={s}")
        axes[0].axhline(res["muon_ns5"], color="gray", lw=0.8)
    axes[0].set(xlabel="refresh period k (steps between eigendecompositions)",
                ylabel=r"alignment $\langle X,O\rangle / (\|X\|_*\|O\|_2)$",
                title="Direction quality (gray: Muon NS-5)")
    axes[0].legend(fontsize=6, ncol=2)

    # ---- figure: alignment over time (sawtooth) for one layer
    name = TRACKED[1]
    seq = seqs[name][burn:]
    for s in ss:
        a, _, _ = replay_ws(seq, 20, s)
        axes[1].plot(a, label=f"WS-Muon k=20, s={s}")
    axes[1].plot([alignment(G, newton_schulz(G, 5).float()) for G in seq], color="gray", label="Muon NS-5")
    axes[1].set(xlabel="step", ylabel="alignment", title=f"{name.replace('.weight', '')}: staleness within a refresh period")
    axes[1].legend(fontsize=7)

    # ---- figure: 1 - alignment vs cache residual (order of the error)
    for s, mk in zip(ss, ["o", "s", "^"]):
        r_all, gap_all = [], []
        for nm in TRACKED:
            sq = seqs[nm][burn:]
            ex = np.array([alignment(G, polar_svd(G)) for G in sq])
            a, _, r = replay_ws(sq, 50, s)
            mask = r > 0
            r_all.append(r[mask])
            gap_all.append(np.maximum(ex[mask] - a[mask], 1e-9))
        r_all, gap_all = np.concatenate(r_all), np.concatenate(gap_all)
        slope = np.polyfit(np.log(r_all), np.log(gap_all), 1)[0]
        axes[2].loglog(r_all, gap_all, mk, ms=2, alpha=0.4, label=f"s={s} (slope {slope:.2f})")
        summary.setdefault("error_order_slope", {})[f"s{s}"] = float(slope)
    axes[2].set(xlabel=r"cache residual $\|P X^\top X P/c - I\|_F/\sqrt{n}$",
                ylabel="alignment(exact) - alignment(WS)", title="Error order vs. staleness")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "ws_muon_alignment.pdf"))
    fig.savefig(os.path.join(args.out, "ws_muon_alignment.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("error-order slopes:", summary["error_order_slope"])
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
