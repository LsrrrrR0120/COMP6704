"""Toy experiments for the "steepest descent under a norm" view of optimizers.

  1. duality check:   argmax_{||D||<=1} <G, D> and its value (dual norm) for l2 / l_inf / spectral
  2. Newton-Schulz:   singular-value map, error vs. iterations, cost vs. SVD
  3. 2-D quadratic:   trajectories of GD / heavy ball / signGD / Adam / Newton (rotation dependence)
  4. matrix least squares with ill-conditioned inputs: GD, Adam, Signum, Shampoo, Muon (NS / exact SVD)

Usage:  python -m experiments.toy_norms [--out figures/toy] [--device cuda]
"""
import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from optimizers import build_optimizer
from optimizers.orthogonalize import CUBIC_COEFFS, QUINTIC_COEFFS, newton_schulz, polar_svd


# ----------------------------------------------------------------------------- 1. duality
def duality_check(device):
    torch.manual_seed(0)
    G = torch.randn(64, 32, device=device)
    U, S, Vh = torch.linalg.svd(G, full_matrices=False)
    rows = []
    for name, D, dual in [
        ("l2 (Frobenius)", G / G.norm(), G.norm()),
        ("l_inf (max-entry)", torch.sign(G), G.abs().sum()),
        ("spectral", U @ Vh, S.sum()),
    ]:
        rows.append(dict(norm=name, inner_product=(G * D).sum().item(), dual_norm=dual.item()))
    return rows


# ----------------------------------------------------------------------------- 2. Newton-Schulz
def ns_scalar_map(sigma, coeffs, steps):
    a, b, c = coeffs
    x = sigma.copy()
    for _ in range(steps):
        x = a * x + b * x ** 3 + c * x ** 5
    return x


def newton_schulz_study(device, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sigma = np.linspace(0, 1, 1000)
    for coeffs, label, ls in [(QUINTIC_COEFFS, "quintic (Muon)", "-"), (CUBIC_COEFFS, "cubic", "--")]:
        for k in (1, 3, 5):
            axes[0].plot(sigma, ns_scalar_map(sigma, coeffs, k), ls, label=f"{label}, {k} steps")
    axes[0].axhline(1, color="k", lw=0.5)
    axes[0].set(xlabel=r"input singular value $\sigma$", ylabel=r"output $\phi^k(\sigma)$",
                title="Newton-Schulz singular-value map")
    axes[0].legend(fontsize=7)

    # Error to the exact polar factor on a gradient-like matrix (fast-decaying spectrum).
    torch.manual_seed(0)
    m, n = 512, 256
    U, _ = torch.linalg.qr(torch.randn(m, n, device=device))
    V, _ = torch.linalg.qr(torch.randn(n, n, device=device))
    G = U @ torch.diag(torch.logspace(0, -3, n, device=device)) @ V.T
    exact = polar_svd(G)
    steps = list(range(0, 16))
    for coeffs, label, dtype in [(QUINTIC_COEFFS, "quintic, bf16", torch.bfloat16),
                                 (QUINTIC_COEFFS, "quintic, fp32", torch.float32),
                                 (CUBIC_COEFFS, "cubic, fp32", torch.float32)]:
        errs = [((newton_schulz(G, k, coeffs, dtype=dtype).float() - exact).norm() / exact.norm()).item() for k in steps]
        axes[1].semilogy(steps, errs, "o-", ms=3, label=label)
    axes[1].set(xlabel="iterations", ylabel=r"$\|X_k - UV^\top\|_F / \|UV^\top\|_F$", title="Distance to polar factor")
    axes[1].legend(fontsize=8)

    # Wall-clock: 5 NS steps (bf16) vs. SVD (fp32), square matrices.
    sizes, t_ns, t_svd = [128, 256, 512, 1024, 2048], [], []
    for s in sizes:
        A = torch.randn(s, s, device=device)
        for fn, store in [(lambda: newton_schulz(A, 5), t_ns), (lambda: polar_svd(A), t_svd)]:
            fn()
            if device.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(5):
                fn()
            if device.type == "cuda":
                torch.cuda.synchronize()
            store.append((time.perf_counter() - t) / 5 * 1e3)
    axes[2].loglog(sizes, t_ns, "o-", label="Newton-Schulz (5 steps, bf16)")
    axes[2].loglog(sizes, t_svd, "s-", label="SVD (fp32)")
    axes[2].set(xlabel="matrix size n (n x n)", ylabel="time [ms]", title="Cost of orthogonalization")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "newton_schulz.pdf"))
    fig.savefig(os.path.join(out, "newton_schulz.png"), dpi=150)
    plt.close(fig)
    return dict(sizes=sizes, ns_ms=t_ns, svd_ms=t_svd)


# ----------------------------------------------------------------------------- 3. 2-D quadratic
def quadratic_trajectories(out, iters=60):
    """f(x) = 0.5 x^T H x with H = R diag(1, 20) R^T. Sign/Adam methods are not
    rotation-invariant, so their behavior changes with the rotation angle."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, angle in zip(axes, (0.0, np.pi / 6)):
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        H = R @ np.diag([1.0, 20.0]) @ R.T
        f = lambda x: 0.5 * x @ H @ x
        x0 = np.array([-2.0, 1.5])
        methods = dict(
            GD=dict(lr=0.09),
            HeavyBall=dict(lr=0.03, beta=0.8),
            signGD=dict(lr=0.08),
            Adam=dict(lr=0.15, b1=0.9, b2=0.999),
            Newton=dict(lr=0.5),
        )
        for name, hp in methods.items():
            x, m, v, path = x0.copy(), np.zeros(2), np.zeros(2), [x0.copy()]
            for t in range(1, iters + 1):
                g = H @ x
                if name == "GD":
                    x = x - hp["lr"] * g
                elif name == "HeavyBall":
                    m = hp["beta"] * m + g
                    x = x - hp["lr"] * m
                elif name == "signGD":
                    x = x - hp["lr"] * np.sign(g) * (0.97 ** t)  # decaying step: sign GD needs it
                elif name == "Adam":
                    m = hp["b1"] * m + (1 - hp["b1"]) * g
                    v = hp["b2"] * v + (1 - hp["b2"]) * g ** 2
                    x = x - hp["lr"] * (m / (1 - hp["b1"] ** t)) / (np.sqrt(v / (1 - hp["b2"] ** t)) + 1e-8) * (0.97 ** t)
                else:
                    x = x - hp["lr"] * np.linalg.solve(H, g)
                path.append(x.copy())
            path = np.array(path)
            ax.plot(path[:, 0], path[:, 1], ".-", ms=3, lw=1, label=f"{name} (f={f(path[-1]):.1e})")
        xs = np.linspace(-2.5, 2.5, 200)
        X, Y = np.meshgrid(xs, xs)
        Z = 0.5 * (H[0, 0] * X ** 2 + 2 * H[0, 1] * X * Y + H[1, 1] * Y ** 2)
        ax.contour(X, Y, Z, levels=np.logspace(-2, 2, 15), colors="gray", linewidths=0.4)
        ax.set(title=f"rotation {np.degrees(angle):.0f}°, cond(H)=20", aspect="equal", xlim=(-2.5, 2.5), ylim=(-2.5, 2.5))
        ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "quadratic_trajectories.pdf"))
    fig.savefig(os.path.join(out, "quadratic_trajectories.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- 4. matrix least squares
class MatrixModel(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(d_out, d_in))

    def forward(self, X):
        return X @ self.W.T


def matrix_least_squares(device, out, iters=300):
    """min_W 1/(2N) ||X W^T - Y||_F^2 with input covariance condition number 1e4."""
    torch.manual_seed(0)
    N, d_in, d_out = 2048, 64, 32
    Q, _ = torch.linalg.qr(torch.randn(d_in, d_in, device=device))
    X = torch.randn(N, d_in, device=device) @ torch.diag(torch.logspace(0, -2, d_in, device=device)) @ Q.T
    W_star = torch.randn(d_out, d_in, device=device)
    Y = X @ W_star.T + 0.01 * torch.randn(N, d_out, device=device)
    W_opt = torch.linalg.lstsq(X, Y).solution.T
    f_opt = (0.5 / N * (X @ W_opt.T - Y).pow(2).sum()).item()

    configs = {
        "GD": ("sgd", {}),
        "Momentum": ("momentum", {}),
        "Adam": ("adam", {}),
        "Signum": ("signum", {}),
        "Shampoo": ("shampoo", dict(precond_freq=1, shampoo_beta=0.95)),
        "Muon (NS-5)": ("muon", dict(scale_mode="spectral")),
        "Spectral GD (exact SVD)": ("muon", dict(orth_method="svd", momentum=0.0, nesterov=False)),
    }
    lrs = np.logspace(-4, 1, 21)
    results = {}
    for label, (name, kw) in configs.items():
        best = None
        for lr in lrs:
            model = MatrixModel(d_in, d_out).to(device)
            opt = build_optimizer(name, model, lr=float(lr), **kw)
            losses = []
            for t in range(iters):
                opt.zero_grad(set_to_none=True)
                loss = 0.5 / N * (model(X) - Y).pow(2).sum()
                loss.backward()
                losses.append(loss.item() - f_opt)
                if not np.isfinite(losses[-1]) or losses[-1] > 1e6:
                    break
                # cosine decay so that fixed-step-size methods (sign / spectral) can converge
                for g in opt.param_groups:
                    g["lr"] = g["initial_lr"] * 0.5 * (1 + np.cos(np.pi * t / iters))
                opt.step()
            if len(losses) == iters and np.isfinite(losses[-1]):
                score = np.mean(losses[-10:])
                if best is None or score < best[0]:
                    best = (score, float(lr), losses)
        results[label] = dict(lr=best[1], final_gap=best[0], curve=best[2])
        print(f"  {label:28s} best lr={best[1]:.2e}  final f-f*={best[0]:.3e}")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for label, r in results.items():
        ax.semilogy(np.maximum(r["curve"], 1e-12), label=f"{label} (lr={r['lr']:.1e})")
    ax.set(xlabel="iteration", ylabel=r"$f(W_t) - f^\star$", title="Matrix least squares, cond(X^T X)=1e4")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "matrix_least_squares.pdf"))
    fig.savefig(os.path.join(out, "matrix_least_squares.png"), dpi=150)
    plt.close(fig)
    return {k: dict(lr=v["lr"], final_gap=v["final_gap"]) for k, v in results.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/toy")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)

    summary = {}
    print("[1] duality check")
    summary["duality"] = duality_check(device)
    for r in summary["duality"]:
        print(f"  {r['norm']:20s} <G,D*>={r['inner_product']:.4f}  dual norm={r['dual_norm']:.4f}")
    print("[2] Newton-Schulz study")
    summary["newton_schulz_timing"] = newton_schulz_study(device, args.out)
    print("[3] 2-D quadratic trajectories")
    quadratic_trajectories(args.out)
    print("[4] matrix least squares")
    summary["matrix_least_squares"] = matrix_least_squares(device, args.out)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"figures and summary written to {args.out}")


if __name__ == "__main__":
    main()
