# M4 — paper vs code contract (Zero-Leakage Reconstruction Routing)

**Date: 2026-09-03. Source reconciliation only. No M4 run exists or is authorised.**

Source: N. Kermiche, *Modular Continual Learning via Zero-Leakage Reconstruction Routing and
Autonomous Task Discovery*, arXiv:2604.14375 (single author, Western Digital).
Official code: `github.com/norikermiche-123/Modular_Continual_Learning` — seven standalone
scripts, two commits both dated 2026-03-29, 0 stars, no tests, no configs, no README-declared
mapping from script to paper table.

The earlier audit recorded that a discrepancy existed. This document fixes exactly what
differs, with primary evidence from both sides.

## Side-by-side

| Component | Paper | Official code path A | Official code path B | Reconciled? |
| --- | --- | --- | --- | --- |
| Novelty threshold | `τ_novelty^(j) = μ_cal^(j) + max(3σ_cal^(j), m)` (Eq. 8) | `NOVELTY_TAU = 0.15` (`unified_cl_framework.py`) | `novelty_threshold = 0.5` (`autonomous_task_discovery.py`); `NOVELTY_TAU = 0.05` (`scaling_task_retrieval.py`) | **No** — three different fixed constants; none is computed |
| Calibration rule | measure `μ_cal`, `σ_cal` of reconstruction error on a **holdout validation split** drawn from the Transient Task Session and purged at commitment | none — no `μ`/`σ` calibration implemented in any script | none | **No** — the paper's calibration mechanism is absent from the code |
| Minimum margin `m` | "latent sparsity floor", ensures a minimum semantic distance even as `σ_cal → 0` | absent | absent | **No** |
| Bottleneck size | `k = 12` optimal by sweep (Table 3: k=4 → 3.67×, **k=12 → 203.78×**, k=32 → 174.23×, k=64 → 176.47× discrimination) | `BN_K = 12` (`realistic_stress_test.py`, `scaling_task_retrieval.py`, `unified_cl_framework.py`); `tbae_latent_dim = 12` (`simultaneous_split_mnist.py`) | `nn.Linear(64, 16)  # 16-D Bottleneck` (`autonomous_task_discovery.py`) | **Partly** — 12 dominates but the autonomous-discovery script uses 16 |
| Commitment gate | Minimum Viable Manifold: provisional router reaches stable reconstruction error **and** Teacher reaches target accuracy **over a sustained K-batch period**; then freeze expert + router, purge raw data, release Teacher | a `[Commitment] ... frozen. Raw data purged.` print in `simultaneous_split_mnist.py`; no MVM condition, no sustained-K-batch test found | absent elsewhere | **No** — the gate is narrated, not implemented |
| Module creation | new Expert–Router pair instantiated when the hard gate fires | `if best_mse < NOVELTY_TAU:` → reuse, else build expert | same | **Partly** — creation exists; its trigger is a fixed constant, not the paper's calibrated gate |
| Routing (inference) | **Contrastive Soft Routing**, `w_i = exp(−ε_i(h)·s) / Σ_j exp(−ε_j(h)·s)` (Eq. 6), with OOD rejection when all familiarity thresholds are exceeded; paper explicitly rejects hard routing as "fragile" | hard `argmin` reconstruction error with a threshold test | same | **No** — the code implements exactly the routing the paper argues against |
| Training-time task info | autonomous discovery; no task IDs | scripts iterate over explicitly named tasks | same | **Partly** — script structure supplies boundaries the paper claims to discover |
| Inference-time task info | none | none | none | **Yes** |
| Benchmark / model | Split-MNIST (2 tasks) for vision; synthetic "Crowded Manifold" 4096-D dataset simulating LLaMA-3 embeddings for NLP | `simultaneous_split_mnist.py`, `split_mnist_full_ablation.py`; `routing_ablation_study.py` for the 4096-D sweep | — | **Yes** |
| Metric definitions | "Task A Retention after Task B"; routing accuracy; discrimination ratio (MSE Task B ÷ MSE Task A) | consistent with the tables | — | **Yes** |

## Two provenance-labelled candidates

Neither is adopted; both are defined so that any future work states which it means.

**`M4-PAPER-CONTRACT`** — dynamic threshold `μ_cal + max(3σ_cal, m)` calibrated on a purged
holdout split; MVM commitment gate over a sustained K-batch period; Contrastive Soft Routing
with OOD rejection at inference; bottleneck `k = 12`. `m`, the sustained-batch count `K`, the
stability criterion for "stable reconstruction error", the Teacher target accuracy and the
sensitivity `s` are **all unspecified numerically** and would be `OURS`.

**`M4-CODE-CONTRACT`** — fixed novelty threshold; no calibration; no MVM gate; hard `argmin`
routing with a threshold test; bottleneck 12 (16 in the discovery script). The threshold
itself is **not single-valued** across the released scripts (0.05 / 0.15 / 0.5), so even this
contract requires an arbitrary choice among the author's own files.

## Can either be called a reproduction of the published result?

**No, in both directions.**

- `M4-PAPER-CONTRACT` cannot be called a reproduction because the numbers in the paper were
  produced by scripts that do not implement it. Reconstructing the paper's mechanism and
  comparing to the paper's tables would compare two different algorithms.
- `M4-CODE-CONTRACT` cannot be called a reproduction of the *method* because the method as
  described — calibrated novelty detection, a commitment gate, soft contrastive routing — is
  the paper's actual contribution and is absent. It also fails to be well-defined, because
  the released scripts disagree with each other on the threshold.

Choosing between them by which better matches the reported numbers is explicitly forbidden
by the evidence-discipline rule and would in any case be circular.

## Second, independent disqualification: scope

Even a perfect reconstruction would adjudicate:

- a **2-task** Split-MNIST retention figure (99.42% vs Experience Replay 95.10%);
- a routing-representation ablation (96.10% best routing accuracy);
- a bottleneck sweep on a **synthetic** dataset built to simulate LLaMA-3 embedding geometry.

There is no long-sequence continual-learning benchmark result. Layer B exists to ask whether
a published gain survives attribution controls; here there is no gain of that kind to test.

## Disposition

**`BLOCKED_CODE_PAPER_CONFLICT`.** Native published-gain adjudication for M4 is blocked, on
two independent grounds. M4 also remains blocked for Layer A, since a standardized transplant
would have to pick one of the conflicting contracts.

This is recorded as a source-fidelity finding, not a criticism of the work's ideas: the
Tight-Bottleneck discrimination sweep is a clean, reusable observation, and the paper's
routing formulation may well be the better one. It simply cannot be attributed to the
released results.
