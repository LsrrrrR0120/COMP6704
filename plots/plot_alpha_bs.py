"""Spectral power alpha x batch size: best-over-lr metric per (alpha, batch size).

Reads the sweeps of configs/{cifar,enwik8}_alpha_bs.yaml, whose labels look like
"a0.25_bs64", "muon_bs64" and "adaptive_s1_bs64".

Usage:
  python -m plots.plot_alpha_bs results/cifar_alpha_bs --out figures/cifar_alpha_bs
  python -m plots.plot_alpha_bs results/enwik8_alpha_bs --out figures/enwik8_alpha_bs
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(root):
    """{(method, bs): {lr: record}} for finished runs."""
    runs = defaultdict(dict)
    for f in glob.glob(os.path.join(root, "*", "*.json")):
        m = re.fullmatch(r"(.+)_bs(\d+)", os.path.basename(os.path.dirname(f)))
        with open(f) as fh:
            r = json.load(fh)
        if m and r.get("status") == "finished":
            runs[(m.group(1), int(m.group(2)))][r["config"]["lr"]] = r
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runs = load(args.root)
    some = next(iter(next(iter(runs.values())).values()))
    metric = "test_acc" if "test_acc" in some["eval"][-1] else "val_loss"
    higher = metric == "test_acc"
    pick = max if higher else min

    best = {k: pick((r["eval"][-1][metric], lr) for lr, r in v.items()) for k, v in runs.items()}
    bss = sorted({bs for _, bs in best})
    alphas = sorted(float(m[1:]) for m in {m for m, _ in best} if re.fullmatch(r"a-?[\d.]+", m))
    others = sorted({m for m, _ in best if not re.fullmatch(r"a-?[\d.]+", m)})

    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(bss)))
    for c, bs in zip(colors, bss):
        al = [a for a in alphas if (f"a{a:g}", bs) in best]
        ys = [best[(f"a{a:g}", bs)][0] for a in al]
        axes[0].plot(al, ys, "o-", color=c, label=f"batch {bs}")
        for m, mk in zip(others, "s^vD"):
            if (m, bs) in best:
                axes[0].plot(alphas[0] - 0.06 if m == "muon" else alphas[-1] + 0.06, best[(m, bs)][0], mk, color=c, mfc="none",
                             label=m if bs == bss[0] else None)
        # regret of each alpha relative to the best alpha at this batch size
        ref = pick(ys)
        axes[1].plot(al, [abs(y - ref) or np.nan for y in ys], "o-", color=c, label=f"batch {bs}")
        for a, y in zip(al, ys):
            rows.append(dict(bs=bs, method=f"a{a:g}", metric=y, lr=best[(f"a{a:g}", bs)][1]))
        for m in others:
            if (m, bs) in best:
                rows.append(dict(bs=bs, method=m, metric=best[(m, bs)][0], lr=best[(m, bs)][1]))
    axes[0].set(xlabel=r"spectral power $\alpha$  (0 = polar, 1 = normalized momentum)",
                ylabel=f"final {metric} (best lr)", title="Best-lr result per power and batch size")
    axes[0].legend(fontsize=7)
    axes[1].set(xlabel=r"$\alpha$", ylabel=f"|{metric} - best over $\\alpha$|", yscale="log",
                title=r"Cost of moving away from the best $\alpha$")
    axes[1].legend(fontsize=7)

    # adaptive variants: estimated SNR and chosen alpha over training
    for m, ls in zip([m for m in others if m.startswith("adaptive")], ["-", "--"]):
        for c, bs in zip(colors, bss):
            if (m, bs) not in best:
                continue
            ev = runs[(m, bs)][best[(m, bs)][1]]["eval"]
            ev = [e for e in ev if "log10_snr_mean" in e]
            x = np.array([e["step"] for e in ev], float)
            axes[2].plot(x / x.max(), [e["log10_snr_mean"] for e in ev], ls, color=c,
                         label=f"{m}, batch {bs}")
    axes[2].set(xlabel="fraction of training", ylabel=r"layer-mean $\log_{10}$ SNR of $X$",
                title="Online SNR estimate (adaptive runs, best lr)")
    axes[2].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "alpha_vs_batch.pdf"))
    fig.savefig(os.path.join(args.out, "alpha_vs_batch.png"), dpi=150)

    with open(os.path.join(args.out, "alpha_vs_batch.csv"), "w") as f:
        f.write("batch_size,method,best_lr,final_" + metric + "\n")
        for r in rows:
            f.write(f"{r['bs']},{r['method']},{r['lr']:g},{r['metric']:.4f}\n")
    print(f"{'method':14s}" + "".join(f"  bs{bs:<6d}" for bs in bss))
    for m in [f"a{a:g}" for a in alphas] + others:
        print(f"{m:14s}" + "".join(f"  {best[(m, bs)][0]:.4f}  " if (m, bs) in best else "    -     " for bs in bss))
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
