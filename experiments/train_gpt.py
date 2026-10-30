"""Train a small GPT on a character-level corpus with one optimizer.

Example:
  python -m experiments.train_gpt --optimizer adamw --lr 1e-3 --steps 3000 \
      --out results/gpt_shakespeare/adamw/lr0.001_seed0.json
"""
import argparse
import json
import math

import torch

from common.data import CORPORA, CharDataset
from common.schedule import lr_factor, set_lr
from common.utils import CudaStepTimer, RunLogger, count_params, get_device, global_grad_norm, set_seed
from models import GPT
from optimizers import OPTIMIZER_NAMES, build_optimizer


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizer", required=True, choices=OPTIMIZER_NAMES)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--aux-lr", type=float, default=None, help="AdamW lr of non-matrix params (Muon/Shampoo)")
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--opt-kwargs", type=json.loads, default={})
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--n-embd", type=int, default=384)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64, help="sequences per optimizer step")
    ap.add_argument("--grad-accum", type=int, default=1, help="micro-batches per step (batch-size must divide)")
    ap.add_argument("--eval-batch-size", type=int, default=64, help="fixed so that val loss is comparable across batch sizes")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--schedule", default="cosine", choices=["cosine", "linear", "constant", "wsd"])
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--grad-clip", type=float, default=0.0, help="0 disables clipping")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default="shakespeare", choices=sorted(CORPORA))
    ap.add_argument("--data", default=None, help="corpus path (defaults to the dataset's path under data/)")
    ap.add_argument("--download", action="store_true", help="download the corpus if it is missing")
    ap.add_argument("--synthetic", action="store_true", help="random characters instead of a corpus (smoke test)")
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--out", required=True)
    return ap.parse_args(argv)


@torch.no_grad()
def estimate_loss(model, ds, args, use_bf16):
    model.eval()
    g = torch.Generator(device=ds.val.device).manual_seed(1234)  # same eval batches for every run
    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(args.eval_iters):
            x, y = ds.get_batch(split, args.eval_batch_size, args.block_size, generator=g)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                losses.append(model(x, y).item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    device = get_device()
    use_bf16 = device.type == "cuda" and not args.no_bf16

    default_path, url = CORPORA[args.dataset]
    ds = CharDataset(args.data or default_path, device, synthetic=args.synthetic, download=args.download, url=url)
    model = GPT(ds.vocab_size, args.block_size, args.n_layer, args.n_head, args.n_embd).to(device)
    opt = build_optimizer(args.optimizer, model, args.lr, weight_decay=args.wd,
                          aux_prefixes=GPT.AUX_PREFIXES, aux_lr=args.aux_lr, **args.opt_kwargs)
    config = dict(vars(args), n_params=count_params(model), vocab_size=ds.vocab_size,
                  device=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")
    logger = RunLogger(args.out, config)
    print(f"[{args.optimizer} lr={args.lr}] {config['n_params'] / 1e6:.2f}M params, {args.steps} steps")
    # The half-batch SNR estimator (SpectralMuon alpha="snr") needs the gradient of the first
    # half of the micro-batches; an odd number of micro-batches is split once more.
    half_batch = getattr(opt, "needs_half_batch", False)
    accum = args.grad_accum * (2 if half_batch and args.grad_accum % 2 else 1)

    status = "finished"
    opt_timer, step_timer = CudaStepTimer(), CudaStepTimer()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            losses = estimate_loss(model, ds, args, use_bf16)
            logger.log_eval(step=step, train_loss=losses["train"], val_loss=losses["val"],
                            opt_step_ms=opt_timer.mean_ms(), train_step_ms=step_timer.mean_ms(),
                            **(opt.adaptive_report() if hasattr(opt, "adaptive_report") else {}))
            logger.save()
            print(f"  step {step:5d}  train {losses['train']:.4f}  val {losses['val']:.4f}")
            if step == args.steps:
                break

        set_lr(opt, lr_factor(step, args.steps, args.warmup_steps, args.schedule))
        step_timer.start()
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        for i in range(accum):
            x, y = ds.get_batch("train", args.batch_size // accum, args.block_size)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                micro_loss = model(x, y) / accum
            micro_loss.backward()
            loss = loss + micro_loss.detach()
            if half_batch and i == accum // 2 - 1:
                opt.mark_half()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        log_now = step % args.log_every == 0
        gnorm = global_grad_norm(model) if log_now else None
        opt_timer.start()
        opt.step()
        opt_timer.stop()
        step_timer.stop()

        if log_now:
            lv = loss.item()
            if not math.isfinite(lv) or lv > 1e2:
                status = "diverged"
                print(f"  diverged at step {step}")
                break
            logger.log(step=step, train_loss=lv, grad_norm=gnorm, lr=opt.param_groups[0]["lr"])

    ev = logger.record["eval"]
    times = [e["opt_step_ms"] for e in ev if e["opt_step_ms"] is not None]
    ttimes = [e["train_step_ms"] for e in ev if e["train_step_ms"] is not None]
    summary = dict(peak_mem_mb=torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else None,
                   opt_step_ms=sum(times) / len(times) if times else None,
                   train_step_ms=sum(ttimes) / len(ttimes) if ttimes else None)
    if ev:
        summary.update(final_val_loss=ev[-1]["val_loss"], best_val_loss=min(e["val_loss"] for e in ev))
    logger.finish(status, **summary)


if __name__ == "__main__":
    main()
