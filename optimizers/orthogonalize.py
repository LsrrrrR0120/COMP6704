"""Approximate and exact orthogonalization (polar factor) of a matrix.

For G = U S V^T (thin SVD), the polar factor U V^T is the steepest-descent
direction of a linear function <G, .> under the spectral norm:

    argmax_{||D||_2 <= 1} <G, D> = U V^T,   with value ||G||_* (nuclear norm).

Muon approximates U V^T with a few Newton-Schulz iterations instead of an SVD.
"""
import torch

# Quintic coefficients tuned by Jordan et al. (2024) to push all singular values
# into roughly [0.7, 1.2] in 5 steps (fast but not exactly convergent).
QUINTIC_COEFFS = (3.4445, -4.7750, 2.0315)
# Classical cubic Newton-Schulz: X <- 1.5 X - 0.5 X X^T X (converges to U V^T).
CUBIC_COEFFS = (1.5, -0.5, 0.0)


@torch.no_grad()
def newton_schulz(G, steps=5, coeffs=QUINTIC_COEFFS, eps=1e-7, dtype=torch.bfloat16):
    """Iterate X <- a X + b (X X^T) X + c (X X^T)^2 X on a 2-D (or batched) matrix."""
    assert G.ndim >= 2
    a, b, c = coeffs
    X = G.to(dtype)
    transposed = X.size(-2) > X.size(-1)
    if transposed:  # work with the smaller Gram matrix
        X = X.mT
    # Normalize so that all singular values are <= 1 (Frobenius >= spectral norm).
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


@torch.no_grad()
def polar_svd(G):
    """Exact polar factor U V^T via SVD (float32)."""
    U, _, Vh = torch.linalg.svd(G.float(), full_matrices=False)
    return (U @ Vh).to(G.dtype)


def orthogonalize(G, method="ns", steps=5):
    if method == "ns":
        return newton_schulz(G, steps=steps)
    if method == "ns_cubic":
        return newton_schulz(G, steps=steps, coeffs=CUBIC_COEFFS, dtype=torch.float32)
    if method == "svd":
        return polar_svd(G)
    raise ValueError(f"unknown orthogonalization method: {method}")
