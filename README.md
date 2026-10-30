# Steepest Descent under Non-Euclidean Norms: SGD, Adam, Shampoo, Muon and Spectral Powers

Code for the COMP6704 (PolyU, Fall 2026) individual assignment of LIN Shenrui (25028243R).
Many deep-learning optimizers can be read as steepest descent under a particular norm:

| norm on the update | steepest-descent direction for gradient G = U S Vᵀ | optimizer family |
|---|---|---|
| Frobenius / l2 | G / ‖G‖_F | SGD, Momentum, NAG |
| l∞ (max entry) | sign(G) | signSGD, Signum, Lion, Adam (β→0) |
| spectral (‖·‖₂ of a matrix) | U Vᵀ (polar factor) | Muon; Shampoo without accumulation |
| Schatten-q, q = 1 + 1/α | U S^α Vᵀ | spectral-power Muon (this work) |

The repository implements eleven optimizers behind a single interface, compares them on
CIFAR-10 (ResNet-18) and byte-level enwik8 (6-layer GPT), and studies the spectral power α:
how the best α depends on the batch size (gradient noise), a noise-adaptive α, and a
fixed-point diagnostic of the gradient noise.

## Layout

```
optimizers/          optimizer library
  __init__.py        build_optimizer(name, model, lr, ...) factory + default hyper-parameters
  muon.py            Muon (momentum + Newton-Schulz orthogonalization), AdamW for non-matrix params
  spectral_power.py  spectral-power Muon U S^alpha V^T (exact, batched eigh); alpha="adaptive"
                     sets alpha per layer from an online SNR estimate (the proposed method)
  ws_muon.py         warm-started Muon with a cached inverse square root (negative result)
  shampoo.py         Shampoo with Kronecker factors, periodic eigh inverse roots, Adam grafting
  sign_methods.py    Signum / signSGD and Lion
  orthogonalize.py   Newton-Schulz (quintic / cubic) and exact SVD polar factor
  adamw_aux.py       AdamW update used for embeddings, heads, biases and norms
models/              CIFAR ResNet (resnet18, resnet_small) and a small GPT
common/              GPU-resident data (CIFAR-10, byte-level text), LR schedules, logging, timers
experiments/
  train_cifar.py           one CIFAR-10 run -> JSON log
  train_gpt.py             one GPT run -> JSON log (supports --grad-accum)
  run_sweep.py             grid of runs from a YAML config (resumable)
  toy_norms.py             duality check, Newton-Schulz study, 2-D quadratic, matrix least squares
  spectral_noise_theory.py i.i.d.-noise model of the best spectral power vs. SNR
  noise_diagnostic.py      fixed-point SNR, noise structure and one-step alpha test
  ws_muon_analysis.py      replays real Muon momenta through cached orthogonalizers
configs/             sweep configs (lr grids per optimizer, alpha x batch size, seeds) and smoke tests
plots/
  plot_curves.py       LR sensitivity, best-lr curves (mean ± std), summary table (CSV + LaTeX)
  plot_alpha_bs.py     best-lr metric per (alpha, batch size)
  plot_alpha_seeds.py  3-seed table of Muon vs alpha = 0 vs alpha = -0.25 per batch size
  report_figures.py    the compact figures used in the report (-> ../report/figures)
results/             JSON logs, one file per run
figures/             generated figures
```

Muon, Shampoo and the spectral-power optimizer update hidden-layer matrices only (conv kernels
are flattened to `out × in·k·k`). The input layer, output layer (`conv1`/`fc` for ResNet,
`tok_emb`/`pos_emb`/`head` for GPT) and all 1-D parameters use AdamW with `--aux-lr`.
Weight decay is never applied to 1-D parameters.

## Setup

Tested with Python 3.13, PyTorch 2.14 (CUDA 13.0) on an RTX 5060 Laptop GPU (8 GB).

```bash
pip install -r requirements.txt
```

All commands below are run from this directory. On Windows with Anaconda, the CPU-only
scripts (`toy_norms`, `spectral_noise_theory`, `ws_muon_analysis`) may need
`KMP_DUPLICATE_LIB_OK=TRUE` because PyTorch and NumPy ship separate OpenMP runtimes.

## Reproducing the results

```bash
# 0. smoke tests (random data, ~2 min): every optimizer runs end-to-end
python -m experiments.run_sweep configs/smoke_cifar.yaml
python -m experiments.run_sweep configs/smoke_gpt.yaml

# 1. data (CIFAR-10 is downloaded automatically by the first CIFAR run)
python -m experiments.train_gpt --dataset enwik8 --optimizer adamw --lr 1e-3 --steps 10 --download --out results/tmp.json

# 2. comparison of eleven optimizers: lr sweep (1 seed), then best lr with 3 seeds
python -m experiments.run_sweep configs/cifar_lr_sweep.yaml
python -m experiments.run_sweep configs/gpt_enwik8_lr_sweep.yaml
python -m experiments.run_sweep configs/cifar_final.yaml
python -m experiments.run_sweep configs/gpt_enwik8_final.yaml
for d in cifar_lr_sweep gpt_enwik8_lr_sweep cifar_final gpt_enwik8_final; do
  python -m plots.plot_curves results/$d --out figures/$d; done

# 3. spectral power alpha x batch size (fixed data budget), alpha < 0, lr-grid extensions, 3 seeds
for t in cifar enwik8; do
  for c in alpha_bs alpha_neg alpha_lrext alpha_seeds; do
    python -m experiments.run_sweep configs/${t}_${c}.yaml; done
  python -m plots.plot_alpha_bs results/${t}_alpha_bs --out figures/${t}_alpha_bs; done
python -m plots.plot_alpha_seeds results/cifar_alpha_bs results/enwik8_alpha_bs --out figures/alpha_seeds

# 4. theory and diagnostics
python -m experiments.spectral_noise_theory            # -> figures/theory
python -m experiments.noise_diagnostic --task gpt      # -> figures/noise
python -m experiments.noise_diagnostic --task cifar
python -m experiments.ws_muon_analysis                 # -> figures/ws_muon
python -m experiments.toy_norms                        # -> figures/toy

# 5. report figures
python -m plots.report_figures                         # -> ../report/figures
```

`configs/gpt_lr_sweep.yaml` (Tiny Shakespeare) is kept as a generalization study: every
optimizer overfits it within a few hundred steps, which is why the main GPT experiments use
enwik8 (one run covers less than one epoch).

`run_sweep.py` skips runs whose JSON already says `finished`/`diverged`, so an interrupted
sweep can be resumed by re-running the same command. `--dry-run` prints the commands and
`--only muon,adamw` restricts the sweep to some optimizers.

A single run, for example:

```bash
python -m experiments.train_cifar --optimizer spectral --opt-kwargs '{"alpha": -0.25}' --lr 0.1 --aux-lr 1e-3 --wd 0.05 --batch-size 1024 --epochs 10 --out results/example.json
python -m experiments.train_gpt --dataset enwik8 --optimizer spectral --opt-kwargs '{"alpha": "adaptive"}' --lr 0.02 --aux-lr 3e-3 --steps 1500 --out results/example_gpt.json
```

Each JSON log contains the full config, the training trace (loss, gradient norm, lr every
`--log-every` steps), evaluation points (with the adaptive optimizer's SNR / alpha ranges),
optimizer-step time (CUDA events) and peak memory.
