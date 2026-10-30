"""Model of how the best spectral power alpha depends on gradient noise.

True gradient G = U diag(theta) V^T with a power-law spectrum theta_i ~ i^(-gamma)
(heavy-tailed, as observed for real momenta). The optimizer only sees X = G + N with
i.i.d. Gaussian noise N whose scale is set by the signal-to-noise ratio
snr = ||G||_F^2 / E||N||_F^2  (snr grows linearly with the batch size).

For an L-smooth loss w.r.t. a norm ||.||, a step -eta D with the best eta decreases the
loss by at least  <G, D>^2 / (2 L ||D||^2).  We therefore score a direction by

    R(D) = E[<G, D>]^2 / E[||D||^2]          (normalised by the noiseless optimum)

for the spectral norm (Muon's geometry) and the Frobenius norm (SGD's geometry), with
D = U_X S_X^alpha V_X^T. Without noise the spectral optimum is alpha = 0 (Muon) and
the Frobenius optimum is alpha = 1 (SGD). The simulation shows that noise does NOT
move the optimum of either pure geometry; it only shrinks the gap between alphas.
With mixed smoothness
    ||D||_mix^2 = kappa ||D||_2^2 + (1 - kappa) ||D||_F^2 / rank
(polar factors have ||D||_2^2 = ||D||_F^2 / rank = 1) the optimum moves from small
alpha (Muon-like) at high SNR towards alpha = 1 (SGD-like) at low SNR: alpha = 1 is
linear in X, so E[D] is proportional to G and the zero-mean noise costs no signal,
whereas the polar map is nonlinear and noise scrambles the singular vectors it
equalizes. Prediction: the best alpha decreases as the batch size grows.

Usage:  python -m experiments.spectral_noise_theory [--out figures/theory]
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from optimizers.spectral_power import spectral_power

ALPHAS = np.round(np.linspace(0, 1, 11), 2)
KAPPAS = [0.02, 0.05, 0.1, 0.2]


def make_signal(m, n, gamma, gen):
    U, _ = torch.linalg.qr(torch.randn(m, n, generator=gen, dtype=torch.float64))
    V, _ = torch.linalg.qr(torch.randn(n, n, generator=gen, dtype=torch.float64))
    theta = torch.arange(1, n + 1, dtype=torch.float64) ** (-gamma)
    G = U @ torch.diag(theta) @ V.T
    return G / G.norm()


def scores(G, snr, trials, gen):
    m, n = G.shape
    num, den_s, den_f = (np.zeros(len(ALPHAS)) for _ in range(3))
    for _ in range(trials):
        N = torch.randn(m, n, generator=gen, dtype=torch.float64) / (m * n) ** 0.5 / snr ** 0.5
        X = G + N
        for i, a in enumerate(ALPHAS):
            D = spectral_power(X, float(a), sigma_floor=1e-6)
            num[i] += (G * D).sum().item()
            den_s[i] += torch.linalg.matrix_norm(D, 2).item() ** 2
            den_f[i] += D.norm().item() ** 2 / n
    num, den_s, den_f = num / trials, den_s / trials, den_f / trials
    best_s = torch.linalg.matrix_norm(G, "nuc").item() ** 2  # noiseless optimum <G,polar G>^2 / 1
    best_f = G.norm().item() ** 2 * n                         # noiseless optimum <G,G>^2/(||G||^2/n)
    rs, rf = num ** 2 / den_s / best_s, num ** 2 / den_f / best_f
    mixed = {k: num ** 2 / (k * den_s + (1 - k) * den_f) for k in KAPPAS}
    return rs, rf, mixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/theory")
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--trials", type=int, default=20)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    gen = torch.Generator().manual_seed(0)

    snrs = np.logspace(-1.5, 2, 15)
    gammas = [0.5, 1.0, 1.5]
    result = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for gamma in gammas:
        G = make_signal(args.m, args.n, gamma, gen)
        best = {k: [] for k in KAPPAS}
        best_s, best_f, curves = [], [], {}
        for snr in snrs:
            rs, rf, mixed = scores(G, snr, args.trials, gen)
            best_s.append(float(ALPHAS[np.argmax(rs)]))
            best_f.append(float(ALPHAS[np.argmax(rf)]))
            for k in KAPPAS:
                best[k].append(float(ALPHAS[np.argmax(mixed[k])]))
            curves[float(snr)] = dict(spectral=rs.tolist(), frobenius=rf.tolist())
        result[gamma] = dict(snr=snrs.tolist(), best_alpha_spectral=best_s, best_alpha_frobenius=best_f,
                             best_alpha_mixed={str(k): v for k, v in best.items()}, curves=curves)
        print(f"gamma={gamma}: best alpha spectral {set(best_s)}, frobenius {set(best_f)}")
        for k in KAPPAS:
            print(f"   kappa={k}: " + ", ".join(f"{s:.2g}->{a:.1f}" for s, a in zip(snrs, best[k])))
            if gamma == 1.0:
                axes[0].semilogx(snrs, best[k], "o-", ms=4, label=rf"$\kappa$={k}")
        if gamma == 1.0:
            for snr in snrs[::3]:
                axes[1].plot(ALPHAS, curves[float(snr)]["spectral"], "o-", ms=3, label=f"snr={snr:.2g}")
                axes[2].plot(ALPHAS, curves[float(snr)]["frobenius"], "o-", ms=3, label=f"snr={snr:.2g}")
    axes[0].set(xlabel=r"SNR $=\|G\|_F^2/\mathbb{E}\|N\|_F^2$ ($\propto$ batch size)",
                ylabel=r"best $\alpha$", title=r"Optimal power, mixed smoothness ($\gamma$=1)")
    axes[0].legend()
    axes[1].set(xlabel=r"$\alpha$", ylabel="normalized guaranteed decrease",
                title=r"Spectral-norm smoothness ($\gamma$=1)")
    axes[2].set(xlabel=r"$\alpha$", ylabel="normalized guaranteed decrease",
                title=r"Frobenius-norm smoothness ($\gamma$=1)")
    axes[1].legend(fontsize=7)
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "optimal_alpha_vs_noise.pdf"))
    fig.savefig(os.path.join(args.out, "optimal_alpha_vs_noise.png"), dpi=150)
    with open(os.path.join(args.out, "optimal_alpha_vs_noise.json"), "w") as f:
        json.dump(result, f)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
