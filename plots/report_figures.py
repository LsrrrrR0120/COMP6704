"""Compact figures for the report (NeurIPS text width 5.5 in), written to ../report/figures.

Usage:  python -m plots.report_figures [--out ../report/figures]

Optimizers are colored by family (Euclidean, diagonal-adaptive, sign, matrix) and told
apart within a family by line style; the palette passes the CVD floor only together with
that secondary encoding, which is why every figure also has a legend.
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

from plots.plot_curves import final, load_runs, mean_curve

FAMILY = {"euclid": "#2a78d6", "adaptive": "#1baf7a", "sign": "#eda100", "matrix": "#e34948"}
STYLE = {  # label -> (display name, family, line style)
    "sgd": ("SGD", "euclid", "-"), "sgd_clip": ("SGD", "euclid", "-"),
    "momentum": ("Momentum", "euclid", "--"), "momentum_clip": ("Momentum", "euclid", "--"),
    "nag": ("NAG", "euclid", ":"), "nag_clip": ("NAG", "euclid", ":"),
    "adagrad": ("Adagrad", "adaptive", ":"), "rmsprop": ("RMSprop", "adaptive", "-."),
    "adam": ("Adam", "adaptive", "--"), "adamw": ("AdamW", "adaptive", "-"),
    "signum": ("Signum", "sign", "-"), "lion": ("Lion", "sign", "--"),
    "shampoo": ("Shampoo", "matrix", "--"), "muon": ("Muon", "matrix", "-"),
}
ORDER = ["sgd", "momentum", "nag", "adagrad", "rmsprop", "adam", "adamw", "signum", "lion", "shampoo", "muon"]
INK, MUTED = "#0b0b0b", "#52514e"


def setup():
    plt.rcParams.update({
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": 0.5, "lines.linewidth": 1.0,
        "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
        "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.4,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
        "legend.frameon": False, "figure.dpi": 150,
    })


def canonical(label):
    return label.replace("_clip", "")


def style(label, emph=("muon", "shampoo")):
    name, fam, ls = STYLE[label]
    return dict(color=FAMILY[fam], ls=ls, lw=1.4 if canonical(label) in emph else 0.9, label=name)


def by_label_lr(root):
    by = defaultdict(lambda: defaultdict(list))
    for r in load_runs(root):
        by[r["label"]][r["config"]["lr"]].append(r)
    return by


def best_lr(by, label, metric, higher):
    scores = {lr: np.nanmean([final(r, metric) for r in rs]) for lr, rs in by[label].items()}
    scores = {k: v for k, v in scores.items() if np.isfinite(v)}
    return (max if higher else min)(scores, key=scores.get)


def labels_for(by):
    """Prefer the clipped variant of the SGD family when it exists (enwik8)."""
    out = []
    for o in ORDER:
        if o + "_clip" in by:
            out.append(o + "_clip")
        elif o in by:
            out.append(o)
    return out


# --------------------------------------------------------------------------- figures
def fig_main(out):
    """Best-lr curves, 3 seeds: CIFAR test accuracy vs epoch, enwik8 val loss vs step and time."""
    cif, gpt = by_label_lr("results/cifar_final"), by_label_lr("results/gpt_enwik8_final")
    fig, ax = plt.subplots(1, 3, figsize=(5.5, 1.85))
    for l in labels_for(cif):
        rs = cif[l][best_lr(cif, l, "test_acc", True)]
        x, m, s = mean_curve(rs, "eval", "test_acc", x_field="epoch")
        ax[0].plot(x, 100 * m, **style(l))
        ax[0].fill_between(x, 100 * (m - s), 100 * (m + s), color=style(l)["color"], alpha=0.15, lw=0)
    ax[0].set(xlabel="epoch", ylabel="test accuracy (%)", ylim=(86, 95), title="(a) CIFAR-10 / ResNet-18")
    for l in labels_for(gpt):
        rs = gpt[l][best_lr(gpt, l, "val_loss", False)]
        x, m, s = mean_curve(rs, "eval", "val_loss")
        ax[1].plot(x, m, **style(l))
        ax[1].fill_between(x, m - s, m + s, color=style(l)["color"], alpha=0.15, lw=0)
        xt, mt, _ = mean_curve(rs, "eval", "val_loss", x_field="wall_time")
        ax[2].plot(xt, mt, **style(l))
    ax[1].set(xlabel="step", ylabel="validation loss (nats/byte)", ylim=(1.0, 1.6), title="(b) enwik8 / GPT")
    ax[2].set(xlabel="wall-clock time (s)", ylim=(1.0, 1.6), title="(c) enwik8, vs. time")
    handles, names = ax[1].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=11, bbox_to_anchor=(0.5, -0.02), handlelength=2.2,
               columnspacing=0.9)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=0.6)
    save(fig, out, "fig_main")


def fig_lr(out):
    """Learning-rate sensitivity (stage-1 sweeps, 1 seed): final metric vs peak lr."""
    fig, ax = plt.subplots(1, 2, figsize=(5.5, 1.75))
    for a, root, metric, title in [(ax[0], "results/cifar_lr_sweep", "test_acc", "(a) CIFAR-10, 10 epochs"),
                                   (ax[1], "results/gpt_enwik8_lr_sweep", "val_loss", "(b) enwik8, 3000 steps")]:
        by = by_label_lr(root)
        for l in labels_for(by):
            lrs = sorted(by[l])
            ys = [np.nanmean([final(r, metric) for r in by[l][lr]]) for lr in lrs]
            ys = [100 * y for y in ys] if metric == "test_acc" else ys
            a.plot(lrs, ys, marker="o", ms=2.2, **style(l))
        a.set(xscale="log", xlabel="peak learning rate", title=title,
              ylabel="test accuracy (%)" if metric == "test_acc" else "validation loss")
        if metric == "test_acc":
            a.set_ylim(80, 94.5)
        else:
            a.set_ylim(1.0, 2.0)
    handles, names = ax[1].get_legend_handles_labels()
    fig.legend(handles, names, loc="center right", ncol=1, bbox_to_anchor=(1.0, 0.5), handlelength=2.2)
    fig.tight_layout(rect=(0, 0, 0.86, 1), w_pad=1.0)
    save(fig, out, "fig_lr")


def alpha_table(root):
    """{(alpha or method, bs): best-over-lr seed-0 metric} of an alpha x batch-size sweep."""
    res = defaultdict(dict)
    metric = None
    for f in glob.glob(os.path.join(root, "*", "*_seed0.json")):
        m = re.fullmatch(r"(.+)_bs(\d+)", os.path.basename(os.path.dirname(f)))
        with open(f) as fh:
            r = json.load(fh)
        if not m or r["status"] != "finished":
            continue
        metric = metric or ("test_acc" if "test_acc" in r["eval"][-1] else "val_loss")
        res[(m.group(1), int(m.group(2)))][r["config"]["lr"]] = r["eval"][-1][metric]
    pick = max if metric == "test_acc" else min
    return metric, {k: pick(v.values()) for k, v in res.items()}


def fig_alpha(out):
    """(a) model prediction of the best alpha vs SNR; (b, c) measured best-lr result vs alpha per batch size."""
    fig, ax = plt.subplots(1, 3, figsize=(5.5, 1.85))
    with open("figures/theory/optimal_alpha_vs_noise.json") as f:
        th = json.load(f)["1.0"]
    blues = ["#9ec5f0", "#5a9be3", "#2a78d6", "#174a8a"]
    for c, (k, ys) in zip(blues, th["best_alpha_mixed"].items()):
        ax[0].plot(th["snr"], ys, marker="o", ms=2, color=c, label=rf"$\kappa$={k}")
    ax[0].set(xscale="log", xlabel=r"SNR $\|G\|_F^2/\mathbb{E}\|N\|_F^2$", ylabel=r"best $\alpha$",
              title="(a) i.i.d.-noise model")
    ax[0].legend(loc="lower left")
    for a, root, title in [(ax[1], "results/cifar_alpha_bs", "(b) CIFAR-10, 10 epochs"),
                           (ax[2], "results/enwik8_alpha_bs", "(c) enwik8, 24.6M tokens")]:
        metric, t = alpha_table(root)
        bss = sorted({bs for _, bs in t})
        alphas = sorted({float(m[1:]) for m, _ in t if re.fullmatch(r"a-?[\d.]+", m)})
        for c, bs in zip(blues[1:], bss):
            al = [x for x in alphas if (f"a{x:g}", bs) in t]
            ys = np.array([t[(f"a{x:g}", bs)] for x in al])
            ys = 100 * ys if metric == "test_acc" else ys
            a.plot(al, ys, marker="o", ms=2.5, color=c, label=f"batch {bs}")
            mu = t[("muon", bs)] * (100 if metric == "test_acc" else 1)
            a.plot([0], [mu], marker="x", ms=4, color=c, ls="none")
        a.axvline(0, color=MUTED, lw=0.5, ls=":")
        a.set(xlabel=r"spectral power $\alpha$", title=title,
              ylabel="test accuracy (%)" if metric == "test_acc" else "validation loss")
        if metric == "val_loss":
            a.set_yscale("log")
            a.set_yticks([1.1, 1.2, 1.4, 1.7, 2.0, 2.5], ["1.1", "1.2", "1.4", "1.7", "2.0", "2.5"])
            a.minorticks_off()
        a.plot([], [], marker="x", color=MUTED, ls="none", label="Muon (NS-5)")
        a.legend(loc="lower left" if metric == "test_acc" else "upper left")
    fig.tight_layout(w_pad=0.6)
    save(fig, out, "fig_alpha")


def fig_noise(out):
    """Fixed-point gradient-noise diagnostic: structure and one-step test, both tasks."""
    res = {t: json.load(open(f"figures/noise/noise_{t}.json")) for t in ("cifar", "gpt")}
    fig, ax = plt.subplots(1, 3, figsize=(5.5, 1.85))
    keys = ["left", "right", "both", "stable_rank_ratio"]
    names = ["left", "right", "both", "stable\nrank"]
    for k, (t, c, name) in enumerate([("cifar", FAMILY["euclid"], "ResNet-18"), ("gpt", FAMILY["matrix"], "GPT")]):
        vals = np.array([[v[q] for q in keys] for v in res[t]["layers"].values()])
        pos = np.arange(len(keys)) + (k - 0.5) * 0.32
        bp = ax[0].boxplot([vals[:, i] for i in range(len(keys))], positions=pos, widths=0.26,
                           patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=0.8),
                           whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6), boxprops=dict(lw=0.6))
        for b in bp["boxes"]:
            b.set_facecolor(c)
            b.set_alpha(0.6)
        ax[0].plot([], [], "s", color=c, alpha=0.6, label=name)
    ax[0].axhline(1, color=MUTED, lw=0.6, ls=":")
    ax[0].set_xticks(range(len(keys)), names)
    ax[0].set(yscale="log", ylabel="ratio to isotropic noise", title="(a) noise structure")
    ax[0].legend(loc="lower left")

    for a, t, title in [(ax[1], "gpt", "(b) one-step test, GPT"), (ax[2], "cifar", "(c) one-step test, ResNet-18")]:
        r = res[t]
        one = r["one_step"]
        full = max(one["full"])
        bs = [k for k in one if k != "full"]
        pick = [bs[0], bs[2], bs[4]]
        for c, b in zip(["#9ec5f0", "#5a9be3", "#174a8a"], pick):
            a.plot(r["alphas"], np.array(one[b]["real"]) / full, "o-", ms=2.5, color=c, label=f"batch {b}")
            a.plot(r["alphas"], np.array(one[b]["iso"]) / full, "x--", ms=3, lw=0.7, color=c)
        a.plot(r["alphas"], np.array(one["full"]) / full, "-", color=INK, lw=1.2, label="mean grad.")
        a.plot([], [], "o-", color=MUTED, ms=2.5, label="real noise")
        a.plot([], [], "x--", color=MUTED, ms=3, lw=0.7, label="isotropic")
        a.set(xlabel=r"spectral power $\alpha$", ylabel="best 1-step decrease (norm.)", title=title, ylim=(0, 1.42), yticks=[0, 0.25, 0.5, 0.75, 1.0])
        a.legend(loc="upper center", fontsize=5, ncol=3, columnspacing=0.6, handlelength=1.6)
    fig.tight_layout(w_pad=0.6)
    save(fig, out, "fig_noise")


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(out, name + ".png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../report/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    setup()
    for f in (fig_main, fig_lr, fig_alpha, fig_noise):
        f(args.out)


if __name__ == "__main__":
    main()
