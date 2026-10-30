"""Run a grid of (optimizer, lr, seed) jobs from a YAML config, one subprocess per run.

Finished runs (status finished/diverged in their JSON) are skipped, so a sweep can
be interrupted and resumed. Usage:
  python -m experiments.run_sweep configs/cifar_resnet18.yaml [--dry-run] [--only muon,adamw]
"""
import argparse
import itertools
import json
import os
import subprocess
import sys

import yaml

from common.utils import run_finished


def jobs_from_config(cfg):
    for opt_name, spec in cfg["optimizers"].items():
        spec = dict(spec)
        lrs = spec.pop("lr")
        lrs = lrs if isinstance(lrs, list) else [lrs]
        seeds = spec.pop("seeds", cfg.get("seeds", [0]))
        label = spec.pop("label", opt_name)  # lets one optimizer appear twice with different settings
        algo = spec.pop("optimizer", opt_name)
        for lr, seed in itertools.product(lrs, seeds):
            args = {**cfg.get("common", {}), **spec, "optimizer": algo, "lr": lr, "seed": seed}
            out = os.path.join(cfg["out_dir"], label, f"lr{lr:g}_seed{seed}.json")
            yield label, args, out


def to_cli(args):
    cli = []
    for k, v in args.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                cli.append(flag)
        elif isinstance(v, dict):
            cli += [flag, json.dumps(v)]
        else:
            cli += [flag, str(v)]
    return cli


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated labels to run")
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    only = set(a.only.split(",")) if a.only else None

    jobs = [j for j in jobs_from_config(cfg) if only is None or j[0] in only]
    todo = [j for j in jobs if not run_finished(j[2])]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} already done, {len(todo)} to run")
    for i, (label, args, out) in enumerate(todo, 1):
        cmd = [sys.executable, "-m", cfg["script"], *to_cli(args), "--out", out]
        print(f"[{i}/{len(todo)}] {' '.join(cmd[2:])}", flush=True)
        if not a.dry_run:
            r = subprocess.run(cmd)
            if r.returncode != 0:
                print(f"  !! exited with code {r.returncode}")


if __name__ == "__main__":
    main()
