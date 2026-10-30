"""AdamW update used for the non-matrix parameters of Muon / Shampoo."""
import math

import torch


@torch.no_grad()
def adamw_update(p, state, lr, betas, eps, weight_decay):
    grad = p.grad
    if "exp_avg" not in state:
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(p)
        state["exp_avg_sq"] = torch.zeros_like(p)
    state["step"] += 1
    beta1, beta2 = betas
    m, v = state["exp_avg"], state["exp_avg_sq"]
    m.lerp_(grad, 1 - beta1)
    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    bc1 = 1 - beta1 ** state["step"]
    bc2 = 1 - beta2 ** state["step"]
    if weight_decay:
        p.mul_(1 - lr * weight_decay)
    denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
    p.addcdiv_(m, denom, value=-lr / bc1)
