import math


def lr_factor(step, total_steps, warmup_steps=0, kind="cosine", final_factor=0.0, decay_frac=0.2):
    """Multiplier applied to every group's initial_lr at `step` (0-based)."""
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    if kind == "constant":
        return 1.0
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    if kind == "cosine":
        return final_factor + (1 - final_factor) * 0.5 * (1 + math.cos(math.pi * progress))
    if kind == "linear":
        return final_factor + (1 - final_factor) * (1 - progress)
    if kind == "wsd":  # warmup-stable-decay: constant, then linear decay over the last decay_frac
        start = 1 - decay_frac
        if progress < start:
            return 1.0
        return final_factor + (1 - final_factor) * (1 - (progress - start) / decay_frac)
    raise ValueError(f"unknown schedule: {kind}")


def set_lr(optimizer, factor):
    for g in optimizer.param_groups:
        g["lr"] = g["initial_lr"] * factor
