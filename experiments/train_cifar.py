"""Train a ResNet on CIFAR-10 with one optimizer and log the optimization trace.

Example:
  python -m experiments.train_cifar --optimizer muon --lr 0.02 --aux-lr 1e-3 --epochs 20 \
      --out results/cifar_resnet18/muon/lr0.02_seed0.json
"""
import argparse
import json
import math

import torch
import torch.nn.functional as F

from common.data import cifar_batches, load_cifar10
from common.schedule import lr_factor, set_lr
from common.utils import CudaStepTimer, RunLogger, count_params, get_device, global_grad_norm, set_seed
from models import build_resnet
from optimizers import OPTIMIZER_NAMES, build_optimizer


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizer", required=True, choices=OPTIMIZER_NAMES)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--aux-lr", type=float, default=None, help="AdamW lr of non-matrix params (Muon/Shampoo)")
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--opt-kwargs", type=json.loads, default={}, help='extra optimizer args as JSON, e.g. \'{"momentum": 0.9}\'')
    ap.add_argument("--model", default="resnet18", choices=["resnet18", "resnet_small"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--schedule", default="cosine", choices=["cosine", "linear", "constant", "wsd"])
    ap.add_argument("--warmup-epochs", type=float, default=1.0)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--synthetic", action="store_true", help="random data instead of CIFAR-10 (smoke test)")
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--out", required=True)
    return ap.parse_args(argv)


@torch.no_grad()
def evaluate(model, x, y, batch_size, use_bf16):
    model.eval()
    loss_sum, correct = 0.0, 0
    for xb, yb in cifar_batches(x, y, 1000, shuffle=False, drop_last=False):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(xb.contiguous(memory_format=torch.channels_last))
        loss_sum += F.cross_entropy(logits.float(), yb, reduction="sum").item()
        correct += (logits.argmax(1) == yb).sum().item()
    model.train()
    return loss_sum / x.size(0), correct / x.size(0)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    device = get_device()
    use_bf16 = device.type == "cuda" and not args.no_bf16
    torch.backends.cudnn.benchmark = True

    data = load_cifar10(args.data_root, device, synthetic=args.synthetic)
    (xtr, ytr), (xte, yte) = data["train"], data["test"]
    model = build_resnet(args.model).to(device).to(memory_format=torch.channels_last)
    opt = build_optimizer(args.optimizer, model, args.lr, weight_decay=args.wd,
                          aux_prefixes=model.AUX_PREFIXES, aux_lr=args.aux_lr, **args.opt_kwargs)

    steps_per_epoch = xtr.size(0) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    warmup = int(args.warmup_epochs * steps_per_epoch)
    config = dict(vars(args), n_params=count_params(model), steps_per_epoch=steps_per_epoch,
                  total_steps=total_steps, device=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")
    logger = RunLogger(args.out, config)
    print(f"[{args.optimizer} lr={args.lr}] {config['n_params'] / 1e6:.2f}M params, {total_steps} steps")

    half_batch = getattr(opt, "needs_half_batch", False)
    step, status, run_loss = 0, "finished", None
    opt_timer, step_timer = CudaStepTimer(), CudaStepTimer()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(args.epochs):
        for xb, yb in cifar_batches(xtr, ytr, args.batch_size, shuffle=True, augment=True):
            set_lr(opt, lr_factor(step, total_steps, warmup, args.schedule))
            step_timer.start()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(xb.contiguous(memory_format=torch.channels_last))
            opt.zero_grad(set_to_none=True)
            if half_batch:
                # Two backward passes through the same full-batch forward (BatchNorm statistics
                # unchanged): first half, mark_half() for the SNR estimator, second half.
                per = F.cross_entropy(logits.float(), yb, label_smoothing=args.label_smoothing, reduction="none")
                h = per.size(0) // 2
                (per[:h].sum() / per.size(0)).backward(retain_graph=True)
                opt.mark_half()
                (per[h:].sum() / per.size(0)).backward()
                loss = per.detach().mean()
            else:
                loss = F.cross_entropy(logits.float(), yb, label_smoothing=args.label_smoothing)
                loss.backward()
            log_now = step % args.log_every == 0
            gnorm = global_grad_norm(model) if log_now else None
            opt_timer.start()
            opt.step()
            opt_timer.stop()
            step_timer.stop()

            if log_now:
                lv = loss.item()
                run_loss = lv if run_loss is None else 0.9 * run_loss + 0.1 * lv
                if not math.isfinite(lv) or lv > 1e3:
                    status = "diverged"
                    break
                logger.log(step=step, epoch=step / steps_per_epoch, train_loss=lv,
                           train_loss_smooth=run_loss, grad_norm=gnorm, lr=opt.param_groups[0]["lr"])
            step += 1
        if status == "diverged":
            print(f"  diverged at step {step}")
            break
        test_loss, test_acc = evaluate(model, xte, yte, 1000, use_bf16)
        logger.log_eval(step=step, epoch=epoch + 1, test_loss=test_loss, test_acc=test_acc,
                        opt_step_ms=opt_timer.mean_ms(), train_step_ms=step_timer.mean_ms(),
                        **(opt.adaptive_report() if hasattr(opt, "adaptive_report") else {}))
        logger.save()
        print(f"  epoch {epoch + 1:3d}  train {run_loss:.4f}  test loss {test_loss:.4f}  acc {test_acc:.4f}")

    ev = logger.record["eval"]
    summary = dict(peak_mem_mb=torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else None)
    if ev:
        summary.update(final_test_acc=ev[-1]["test_acc"], final_test_loss=ev[-1]["test_loss"],
                       best_test_acc=max(e["test_acc"] for e in ev),
                       opt_step_ms=_mean([e["opt_step_ms"] for e in ev]),
                       train_step_ms=_mean([e["train_step_ms"] for e in ev]))
    logger.finish(status, **summary)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


if __name__ == "__main__":
    main()
