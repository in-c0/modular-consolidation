# M6 — compute authorization dossier

**Purpose: make the next execution decision mechanical rather than speculative.**
No scientific run has been executed. This document authorises nothing.

## Measured basis

All figures below rest on measurements taken during the score-free mechanical smoke, on the
target machine, using the official entrypoint at the pinned revision.

| Measurement | Value | How obtained |
| --- | --- | --- |
| Machine | Apple M1 Max, macOS 15.6, 10 cores | host |
| Backend | JAX 0.11.1 **CPU device**, optax 0.2.8, torch 2.14.0 (dataloading only) | `jax.devices()` |
| Gradient step, widths (12,12) | 0.232 ms | jitted microbenchmark, batch 256 |
| Gradient step, widths (24,38) | 0.312 ms | jitted microbenchmark, batch 256 |
| ED SVD per layer | 0.197–0.281 ms | `trigger_act` microbenchmark |
| **Official entrypoint, 2 tasks × 2 epochs** | **21 s wall-clock** | `train.py --config <probe> --seed 0` |
| MNIST load (cached) | 2.9 s | smoke check [6] |

The 21 s figure includes interpreter start, JAX import and compile, dataloading, EWC/Fisher,
evaluation and the un-jitted Python epoch loop — i.e. everything the microbenchmark omits.
Taking ~10 s of that as fixed startup gives **≈2.7 s per epoch** of real work.

## Projected cost of the preregistered run

Binary Split MNIST 2L: 5 tasks, `n1_epochs: 10` + 4 × `n_epochs: 30` = **130 epochs per run**.

| Item | Value |
| --- | --- |
| Benchmark | Binary Split MNIST, 2 hidden layers |
| Tasks | 5 |
| Model | MLP, 784 → h → h → 2, no biases |
| Base parameters | 9 576 (widths 12/12) |
| Expected final parameters | ≈19 255 ± 456 (published), widths ≈24/38 |
| `C-MATCH` static width | 24 → 19 440 params (−364 residual vs seed 0) |
| `C-PAPER` static width | 32 → 26 176 params |
| Primary arms | 7 (`T-FULL`, `T-ED`, `T-FISHER`, `T-COUNT`, `I-RANDOM`, `I-XAVIER`, `C-MATCH`) |
| Reference arm | `C-PAPER` (published baseline, not causal) |
| Seeds | 5 (0–4), paired |
| Repetitions | 1 per (arm, seed) |
| **Total model-training runs** | **7 × 5 = 35 primary, + 5 reference = 40** |
| Epochs per run | 130 |
| **Estimated wall-clock per run** | **≈6 min** (10 s startup + 130 × 2.7 s), rising toward ~10 min on later tasks as widths grow and the evaluation set accumulates |
| **Sequential total** | **≈4–7 h** |
| **With 5 seeds in parallel** (10 cores) | **≈1–1.5 h** |
| Peak RAM | **< 1 GB** — the largest model is ~20 k parameters; MNIST in memory dominates |
| GPU RAM | **none required** |
| Disk / checkpoints | ≈40 runs × (per-task CSV, per-epoch CSV, acc matrix, summary, timing, growth JSONL) ≈ **< 50 MB** |

### Is the Mac sufficient?

**Yes, decisively.** The model is a ~20 k-parameter MLP and the whole preregistered programme
is 40 short CPU runs. The measured 21 s for 4 epochs makes this unambiguous.

**Is an RTX 2080 materially preferable? No.** At batch 256 and this width, a GPU is dominated
by kernel-launch and host-transfer overhead; the workload is far too small to amortise it. The
source README's "~20 GPU-h" figure for its Table 4 covers *all* benchmarks, depths, widths and
the full ablation grid across Permuted/Rotated MNIST and CIFAR — an order of magnitude more
work than the seven-arm, single-benchmark, single-depth set preregistered here. Do not read
that number as applying to this run.

### Control reuse

Controls **cannot** reuse trained checkpoints: every arm changes the training trajectory
itself (trigger, initialization, or static width), so each is an independent training run.
That is already reflected in the 40-run count. The only reuse is the derivation step —
`T-COUNT`'s `k_fixed` and `C-MATCH`'s `hidden_dim` are computed from `T-FULL`'s realised
widths, so `T-FULL` must complete before those two arms are configured.

### Execution order implied by the derivations

1. `T-FULL` × 5 seeds → realised final widths per seed (also the Gate-R arm).
2. Gate R evaluated. If it fails, stop and label `NATIVE REPRODUCTION FAILED`.
3. Derive `T-COUNT.k_fixed` and `C-MATCH.hidden_dim` per seed, freeze, commit.
4. Remaining six arms × 5 seeds.

Step 2 is a real stopping point: roughly 30 minutes of compute answers whether the rest is
worth running at all.

## Secondary / exploratory, not authorised here

Permuted MNIST 2L (10 tasks) and Rotated MNIST 2L as replication; 1-layer variants;
`loss_plateau`, `he_normal`, `nullspace`, `zero`, `vp_zfo`; `static16`/`static64`;
Binary Split CIFAR-10 (needs feature extraction, a different cost class).

## Boundary

**M6 SCIENTIFIC RUN: PREREGISTERED / UNRUN — AWAITING COMPUTE AUTHORIZATION.**
