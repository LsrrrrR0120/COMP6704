import json
import os
import random
import time

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def global_grad_norm(model):
    norms = [p.grad.norm() for p in model.parameters() if p.grad is not None]
    return torch.linalg.vector_norm(torch.stack(norms)).item() if norms else 0.0


class RunLogger:
    """Collects a run's config, history and summary and writes them to one JSON file."""

    def __init__(self, path, config):
        self.path = path
        self.record = dict(config=config, history=[], eval=[], summary={}, status="running")
        self.t0 = time.time()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def log(self, **row):
        row["wall_time"] = time.time() - self.t0
        self.record["history"].append(row)

    def log_eval(self, **row):
        row["wall_time"] = time.time() - self.t0
        self.record["eval"].append(row)

    def finish(self, status="finished", **summary):
        self.record["status"] = status
        self.record["summary"].update(summary, total_time=time.time() - self.t0)
        self.save()

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.record, f)
        os.replace(tmp, self.path)


def run_finished(path):
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            return json.load(f).get("status") in ("finished", "diverged")
    except (json.JSONDecodeError, OSError):
        return False


class CudaStepTimer:
    """Times a code region with CUDA events without forcing a sync each step."""

    def __init__(self, enabled=True):
        self.enabled = enabled and torch.cuda.is_available()
        self.pairs = []

    def start(self):
        if self.enabled:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self.pairs.append([e, None])

    def stop(self):
        if self.enabled:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self.pairs[-1][1] = e

    def mean_ms(self, skip=5):
        """Mean elapsed ms, ignoring the first `skip` (warm-up) measurements; clears the buffer."""
        if not self.enabled or len(self.pairs) <= skip:
            self.pairs = []
            return None
        torch.cuda.synchronize()
        ms = [a.elapsed_time(b) for a, b in self.pairs[skip:]]
        self.pairs = []
        return float(np.mean(ms))
