"""3-seed comparison of alpha = 0, alpha = -0.25 and Muon at every batch size.

For each (method, batch size) the lr with the best seed-mean among those run with 3 seeds
(configs/*_alpha_seeds.yaml) is reported as mean +- std.

Usage:
  python -m plots.plot_alpha_seeds results/cifar_alpha_bs results/enwik8_alpha_bs --out figures/alpha_seeds
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

METHODS = {"muon": "Muon (NS-5)", "a0": r"$\alpha=0$ (exact polar)", "a-0.25": r"$\alpha=-0.25$",
           "snr": r"half-batch adaptive $\alpha$"}


def summarize(root):
    """{method: {bs: (mean, std, lr, n)}} over lrs that have >= 3 finished seeds."""
    vals = defaultdict(list)
    metric = None
    for f in glob.glob(os.path.join(root, "*", "*.json")):
        m = re.fullmatch(r"(.+)_bs(\d+)", os.path.basename(os.path.dirname(f)))
        if not m or m.group(1) not in METHODS:
            continue
        with open(f) as fh:
            r = json.load(fh)
        if r.get("status") != "finished":
            continue
        metric = metric or ("test_acc" if "test_acc" in r["eval"][-1] else "val_loss")
        vals[(m.group(1), int(m.group(2)), r["config"]["lr"])].append(r["eval"][-1][metric])
    higher = metric == "test_acc"
    out = defaultdict(dict)
    for (m, bs, lr), v in vals.items():
        if len(v) < 3:
            continue
        mean = float(np.mean(v))
        cur = out[m].get(bs)
        if cur is None or (mean > cur[0] if higher else mean < cur[0]):
            out[m][bs] = (mean, float(np.std(v)), lr, len(v))
    return metric, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig, axes = plt.subplots(1, len(args.roots), figsize=(5.5 * len(args.roots), 3.8))
    axes = np.atleast_1d(axes)
    tex = []
    for ax, root in zip(axes, args.roots):
        name = os.path.basename(os.path.normpath(root)).replace("_alpha_bs", "")
        metric, res = summarize(root)
        scale = 100 if metric == "test_acc" else 1
        bss = sorted({bs for v in res.values() for bs in v})
        for k, (m, label) in enumerate(METHODS.items()):
            pts = [(bs, *res[m][bs]) for bs in bss if bs in res.get(m, {})]
            if not pts:
                continue
            x = np.array([p[0] for p in pts]) * (1 + 0.04 * (k - 1))
            ax.errorbar(x, [p[1] * scale for p in pts], [p[2] * scale for p in pts], marker="o", capsize=3,
                        label=label)
        ax.set(xscale="log", xlabel="batch size", title=name,
               ylabel="test accuracy (%)" if metric == "test_acc" else "validation loss")
        ax.set_xticks(bss, [str(b) for b in bss])
        ax.legend(fontsize=8)

        print(f"== {name} ({metric}, mean +- std over 3 seeds at the best lr)")
        tex.append(f"% {name}: {metric}")
        for m, label in METHODS.items():
            cells = []
            for bs in bss:
                if bs in res.get(m, {}):
                    mean, std, lr, n = res[m][bs]
                    cells.append(f"{mean * scale:.2f}$\\pm${std * scale:.2f}" if scale == 100
                                 else f"{mean:.4f}$\\pm${std:.4f}")
                    print(f"  {m:8s} bs{bs:<5d} {mean * scale:.4f} +- {std * scale:.4f}  (lr {lr:g}, n={n})")
                else:
                    cells.append("--")
            tex.append(label + " & " + " & ".join(cells) + r" \\")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "alpha_seeds.pdf"))
    fig.savefig(os.path.join(args.out, "alpha_seeds.png"), dpi=150)
    with open(os.path.join(args.out, "alpha_seeds.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
