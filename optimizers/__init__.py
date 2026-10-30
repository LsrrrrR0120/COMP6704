"""Optimizer factory shared by all experiments.

Element-wise optimizers update every parameter with the same rule.
Matrix optimizers (Muon, Shampoo) update hidden-layer matrices with their own
rule and all remaining parameters (input/output layers given by
`aux_prefixes`, plus biases/norm gains) with AdamW.
Weight decay is never applied to 1-D parameters (biases, norm gains).
"""
import torch

from .muon import Muon
from .shampoo import Shampoo
from .sign_methods import Lion, Signum
from .spectral_power import SpectralMuon
from .ws_muon import WSMuon

# Default hyper-parameters per optimizer; anything passed to build_optimizer overrides them.
DEFAULTS = {
    "sgd": dict(momentum=0.0),
    "momentum": dict(momentum=0.9),
    "nag": dict(momentum=0.9, nesterov=True),
    "adagrad": dict(eps=1e-10),
    "rmsprop": dict(alpha=0.99, eps=1e-8),
    "adam": dict(betas=(0.9, 0.999), eps=1e-8),
    "adamw": dict(betas=(0.9, 0.999), eps=1e-8),
    "signsgd": dict(momentum=0.0),
    "signum": dict(momentum=0.9),
    "lion": dict(betas=(0.9, 0.99)),
    "muon": dict(momentum=0.95, nesterov=True, ns_steps=5, orth_method="ns", scale_mode="spectral"),
    "shampoo": dict(betas=(0.9, 0.999), shampoo_beta=0.99, precond_freq=20, max_precond_dim=2048),
    "ws_muon": dict(momentum=0.95, nesterov=True, scale_mode="spectral", refresh_every=10,
                    correction_steps=1, damping=1e-3, refresh_tol=0.5),
    "spectral": dict(momentum=0.95, nesterov=True, scale_mode="spectral", alpha=0.5, sigma_floor=1e-3),
}
ELEMENTWISE = ["sgd", "momentum", "nag", "adagrad", "rmsprop", "adam", "adamw", "signsgd", "signum", "lion"]
MATRIX = ["muon", "shampoo", "ws_muon", "spectral"]
OPTIMIZER_NAMES = ELEMENTWISE + MATRIX


def split_params(model, aux_prefixes=()):
    """Return (hidden matrices, input/output-layer matrices, 1-D params)."""
    hidden, aux, vectors = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2:
            vectors.append(p)
        elif name.startswith(tuple(aux_prefixes)):
            aux.append(p)
        else:
            hidden.append(p)
    return hidden, aux, vectors


def build_optimizer(name, model, lr, weight_decay=0.0, aux_prefixes=(), aux_lr=None, **kwargs):
    """Build optimizer `name`. `aux_lr` is the AdamW lr of non-matrix params for
    Muon/Shampoo (defaults to `lr`). Each param group remembers `initial_lr`
    so that schedules can rescale it."""
    if name not in DEFAULTS:
        raise ValueError(f"unknown optimizer {name}; choose from {OPTIMIZER_NAMES}")
    hp = {**DEFAULTS[name], **kwargs}
    hidden, aux, vectors = split_params(model, aux_prefixes)

    if name in ELEMENTWISE:
        groups = [
            dict(params=hidden + aux, weight_decay=weight_decay),
            dict(params=vectors, weight_decay=0.0),
        ]
    else:
        flag = "use_shampoo" if name == "shampoo" else "use_muon"
        aux_lr = lr if aux_lr is None else aux_lr
        groups = [
            {"params": hidden, "weight_decay": weight_decay, flag: True},
            {"params": aux, "weight_decay": weight_decay, "lr": aux_lr, flag: False},
            {"params": vectors, "weight_decay": 0.0, "lr": aux_lr, flag: False},
        ]
    groups = [g for g in groups if g["params"]]

    if name in ("sgd", "momentum", "nag"):
        opt = torch.optim.SGD(groups, lr=lr, **hp)
    elif name == "adagrad":
        opt = torch.optim.Adagrad(groups, lr=lr, **hp)
    elif name == "rmsprop":
        opt = torch.optim.RMSprop(groups, lr=lr, **hp)
    elif name == "adam":  # coupled L2 regularization
        opt = torch.optim.Adam(groups, lr=lr, **hp)
    elif name == "adamw":
        opt = torch.optim.AdamW(groups, lr=lr, **hp)
    elif name in ("signsgd", "signum"):
        opt = Signum(groups, lr=lr, **hp)
    elif name == "lion":
        opt = Lion(groups, lr=lr, **hp)
    elif name == "muon":
        opt = Muon(groups, lr=lr, **hp)
    elif name == "ws_muon":
        opt = WSMuon(groups, lr=lr, **hp)
    elif name == "spectral":
        opt = SpectralMuon(groups, lr=lr, **hp)
    else:
        opt = Shampoo(groups, lr=lr, **hp)

    for g in opt.param_groups:
        g["initial_lr"] = g["lr"]
    return opt


__all__ = ["build_optimizer", "split_params", "OPTIMIZER_NAMES", "ELEMENTWISE", "MATRIX",
           "Muon", "WSMuon", "SpectralMuon", "Shampoo", "Lion", "Signum"]
