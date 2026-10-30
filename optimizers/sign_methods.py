"""Steepest descent under the l_inf norm: signSGD / Signum and Lion.

argmax_{||d||_inf <= 1} <g, d> = sign(g), so sign-based methods are the
l_inf counterpart of (normalized) gradient descent.
"""
import torch
from torch.optim import Optimizer


class Signum(Optimizer):
    """Signum (Bernstein et al., 2018): x <- x - lr * sign(m), m = EMA of gradients.

    momentum=0 recovers signSGD. Weight decay is decoupled.
    """

    def __init__(self, params, lr=1e-4, momentum=0.9, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, beta, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if beta > 0:
                    state = self.state[p]
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(p)
                    m = state["exp_avg"].lerp_(p.grad, 1 - beta)
                else:
                    m = p.grad
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(torch.sign(m), alpha=-lr)
        return loss


class Lion(Optimizer):
    """Lion (Chen et al., 2023): sign of an interpolation between momentum and gradient."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, (beta1, beta2), wd = group["lr"], group["betas"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)
                m = state["exp_avg"]
                update = torch.sign(m.lerp(p.grad, 1 - beta1))
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)
                m.lerp_(p.grad, 1 - beta2)
        return loss
