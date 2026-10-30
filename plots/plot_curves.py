"""Summarize a sweep directory: LR sensitivity, best-LR curves (mean +- std over seeds), table.

Usage:
  python -m plots.plot_curves results/cifar_lr_sweep --out figures/cifar_lr_sweep
  python -m plots.plot_curves results/gpt_lr_sweep  --out figures/gpt_lr_sweep
The evaluation metric is test_acc (higher is better) for CIFAR runs and val_loss
(lower is better) for GPT runs; it is detected from the logs.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ORDER = ["sgd", "sgd_clip", "momentum", "momentum_clip", "nag", "nag_clip", "adagrad", "rmsprop", "adam", "adamw",
         "signsgd", "signum", "lion", "shampoo", "muon"]


def load_runs(root):
    runs = []
    for path in glob.glob(os.path.join(root, "*", "*.json")):
        with open(path) as f:
            r = json.load(f)
        if r["status"] == "running":
            continue
        r["label"] = os.path.basename(os.path.dirname(path))
        runs.append(r)
    return runs


def metric_of(runs):
    keys = runs[0]["eval"][0].keys() if runs and runs[0]["eval"] else []
    return ("test_acc", True) if "test_acc" in keys else ("val_loss", False)


def final(run, metric):
    return run["eval"][-1][metric] if run["status"] == "finished" and run["eval"] else np.nan


def sort_labels(labels):
    return sorted(labels, key=lambda l: (ORDER.index(l) if l in ORDER else len(ORDER), l))


def mean_curve(runs, key, field, x_field="step"):
    """Average `field` across seeds on the shortest common grid."""
    seqs = [[row for row in r[key] if row.get(field) is not None] for r in runs]
    n = min(len(s) for s in seqs)
    if n == 0:
        return None
    x = np.array([row[x_field] for row in seqs[0][:n]])
    ys = np.array([[row[field] for row in s[:n]] for s in seqs])
    return x, ys.mean(0), ys.std(0)


def smooth(y, k=10):
    if len(y) < k:
        return y
    return np.convolve(y, np.ones(k) / k, mode="valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs = load_runs(args.results)
    if not runs:
        raise SystemExit(f"no finished runs under {args.results}")
    metric, higher = metric_of(runs)
    by = defaultdict(lambda: defaultdict(list))  # label -> lr -> runs
    for r in runs:
        by[r["label"]][r["config"]["lr"]].append(r)
    labels = sort_labels(by)
    colors = {l: plt.cm.tab20(i / max(1, len(labels) - 1)) for i, l in enumerate(labels)}

    # ---- best lr per optimizer
    best = {}
    for l in labels:
        scores = {lr: np.nanmean([final(r, metric) for r in rs]) for lr, rs in by[l].items()}
        valid = {lr: s for lr, s in scores.items() if np.isfinite(s)}
        if valid:
            best[l] = (max if higher else min)(valid, key=valid.get)

    # ---- LR sensitivity
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for l in labels:
        lrs = sorted(by[l])
        ys = [np.nanmean([final(r, metric) for r in by[l][lr]]) for lr in lrs]
        ax.semilogx(lrs, ys, "o-", color=colors[l], label=l, ms=4)
    ax.set(xlabel="peak learning rate", ylabel=f"final {metric}", title="Learning-rate sensitivity")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "lr_sensitivity.pdf"))
    fig.savefig(os.path.join(args.out, "lr_sensitivity.png"), dpi=150)
    plt.close(fig)

    # ---- curves at the best lr: train loss vs step, eval metric vs step, eval metric vs time
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for l, lr in best.items():
        rs = [r for r in by[l][lr] if r["status"] == "finished"]
        c = colors[l]
        tr = mean_curve(rs, "history", "train_loss")
        if tr is not None:
            x, m, _ = tr
            axes[0].plot(x[len(x) - len(smooth(m)):], smooth(m), color=c, label=f"{l} ({lr:g})")
        ev = mean_curve(rs, "eval", metric)
        if ev is not None:
            x, m, s = ev
            axes[1].plot(x, m, color=c, label=l)
            axes[1].fill_between(x, m - s, m + s, color=c, alpha=0.2)
            xt, mt, _ = mean_curve(rs, "eval", metric, x_field="wall_time")
            axes[2].plot(xt, mt, color=c, label=l)
    axes[0].set(xlabel="step", ylabel="train loss (smoothed)", yscale="log", title="Training loss at best lr")
    axes[1].set(xlabel="step", ylabel=metric, title=f"{metric} (mean ± std over seeds)")
    axes[2].set(xlabel="wall-clock time [s]", ylabel=metric, title=f"{metric} vs. time")
    # zoom on the end of training: from the best final value to a bit past the worst one
    finals = [np.nanmean([final(r, metric) for r in by[l][lr]]) for l, lr in best.items()]
    lo, hi = np.nanmin(finals), np.nanmax(finals)
    pad = hi - lo
    for a in axes[1:]:
        a.set_ylim((lo - 0.8 * pad, hi + 0.1 * pad) if higher else (lo - 0.1 * pad, hi + 0.8 * pad))
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "best_curves.pdf"))
    fig.savefig(os.path.join(args.out, "best_curves.png"), dpi=150)
    plt.close(fig)

    # ---- table
    rows = []
    for l, lr in best.items():
        rs = by[l][lr]
        vals = np.array([final(r, metric) for r in rs])
        s = [r["summary"] for r in rs]
        mean = lambda k: np.nanmean([x.get(k) if x.get(k) is not None else np.nan for x in s])
        rows.append(dict(optimizer=l, best_lr=lr, n_seeds=len(rs), metric=metric,
                         mean=np.nanmean(vals), std=np.nanstd(vals),
                         opt_step_ms=mean("opt_step_ms"), train_step_ms=mean("train_step_ms"),
                         peak_mem_mb=mean("peak_mem_mb")))
    rows.sort(key=lambda r: -r["mean"] if higher else r["mean"])
    with open(os.path.join(args.out, "summary.csv"), "w") as f:
        f.write(",".join(rows[0].keys()) + "\n")
        for r in rows:
            f.write(",".join(str(v) for v in r.values()) + "\n")
    with open(os.path.join(args.out, "summary.tex"), "w") as f:
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write(f"Optimizer & best lr & {metric.replace('_', ' ')} & opt.\\ step [ms] & train step [ms] \\\\\n\\midrule\n")
        for r in rows:
            scale = 100 if metric == "test_acc" else 1
            f.write(f"{r['optimizer']} & {r['best_lr']:g} & {scale * r['mean']:.2f} $\\pm$ {scale * r['std']:.2f} "
                    f"& {r['opt_step_ms']:.2f} & {r['train_step_ms']:.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    for r in rows:
        print(f"{r['optimizer']:10s} lr={r['best_lr']:<8g} {metric}={r['mean']:.4f}+-{r['std']:.4f}  "
              f"opt step {r['opt_step_ms']:.2f} ms")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
