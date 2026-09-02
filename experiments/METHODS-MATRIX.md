# Methods-paper experiment matrix

Per owner decisions D3 and D6 (`docs/OWNER-DECISIONS.md`), the primary near-term paper is a
methods/evaluation paper about **attribution** in modular continual learning. It does not
propose a new policy.

**Status: design. No real-model re-analysis has been run.** Nothing below is a claim about
any selected paper; predicted confounds state what the control lattice will test.

## Selection rule

Methods are selected to **span the factor space**, not by popularity or reported score. One
representative per cell is chosen for (a) reproducibility from the paper, (b) a clearly
identifiable allocation/consolidation mechanism, and (c) coverage of a distinct factor.
Selection was fixed before any re-analysis. Additions require a dated amendment.

| # | Method | Factor it exemplifies | Allocation | Consolidation | Routing | Task IDs |
| --- | --- | --- | --- | --- | --- | --- |
| M1 | Fixed LoRA bank + learned router (constructed reference) | routing at fixed capacity | none | none | learned | no |
| M2 | Progressive Networks / DEN family | unbounded expansion | per task | none | task-indexed | yes |
| M3 | MoCL-P — *Learn it or Leave it* (Repl4NLP 2024, arXiv:2406.18708) | composition + **pruning** | per task | prune | composition weights | yes |
| M4 | Zero-Leakage Reconstruction Routing (arXiv:2604.14375) | **task-free discovery**, no consolidation | on novelty | none | reconstruction error | no |
| M5 | MADE-IT — Manifold-Aware Expert Evolution (arXiv:2604.22464) | **merge-based consolidation** | on novelty | merge | training-free, subspace | no |
| M6 | NORACL (arXiv:2604.27031) | **saturation-triggered growth** | on saturation | none | — | no |
| M7 | Latent-LoRA (arXiv:2607.23837) | one adapter per task, **gradient-free routing** | per task | none | GMM over embeddings | no |
| M8 | FLAME (arXiv:2605.09355) | **fixed-size pool**, compression instead of expansion | none | low-rank compression | modality routers | no |

M1 exists because the lattice needs a reference isolating routing without allocation or
consolidation; no selected published method occupies that cell cleanly.

## D6: two evidence layers, not one forced benchmark

The previous requirement for one native-fidelity benchmark across all eight methods is
withdrawn. Cross-method mechanism comparability and attribution of a source paper's result
are different scientific objectives.

### Layer A — standardized mechanism panel

**Primary benchmark: Long Sequence / MTL15.** Use the two established fixed task orders,
task-agnostic inference, and task identity only for scoring and `C-OID`. All tasks are
classification tasks with accuracy as the common metric, making one retention–plasticity
frontier interpretable. The canonical sequence is not modified.

A method moved onto this substrate is a **standardized implementation of its mechanism**,
not a reproduction of the source paper's headline result.

Initial standardized panel:

| # | Standardized mechanism | Included now? | Exact transplant definition still required? |
| --- | --- | --- | --- |
| M1 | fixed LoRA bank + learned task-free router | yes | model/rank/router gated on `plasticity-routing` |
| M2 | grow one LoRA expert at each declared allocation event | yes | replace native task-indexed execution with explicit `C-OID`/task-free variants |
| M3 | compose LoRA experts and prune under the paper's pruning rule | yes | yes — pruning trigger and composition rule must be frozen |
| M4 | novelty allocation using reconstruction-style routing | yes | yes — reconstruction representation/router substrate must be frozen |
| M5 | manifold-aware merge rule | **not yet** | native CLIP-ViT mechanism; LoRA transplant requires independent validation |
| M6 | saturation-triggered LoRA growth | yes | yes — saturation statistic/threshold must be frozen without outcome tuning |
| M7 | per-task LoRA bank with GMM/latent routing | yes | model/rank/embedding representation gated on `plasticity-routing` |
| M8 | fixed-pool low-rank compression | **not yet** | native multimodal MoE mechanism; LoRA transplant requires independent validation |

The standardized panel may therefore begin with M1/M2/M3/M4/M6/M7 once the shared
model/rank/routing substrate is resolved. M5 and M8 do not enter merely to fill cells.

A recurrent-return Long Sequence variant is permitted only as a separately named,
predeclared stress test if returning-task/reinstatement behaviour must be exercised.

### Layer B — native-fidelity re-analysis

Each published method is reconstructed on its original substrate/benchmark as closely as
feasible and then receives the missing attribution controls. Only this layer licenses claims
about whether a published gain survives the protocol. MADE-IT and FLAME remain here unless
and until their standardized LoRA transplants are independently validated.

## What the lattice adds to each

| # | Reported comparison (as published) | Predicted confound the lattice tests | Missing control(s) |
| --- | --- | --- | --- |
| M1 | — | — | reference point |
| M2 | vs fixed-capacity baselines of different size | growth vs routing/capacity | `C-TERM`: fixed bank at realised final size; task-free vs `C-OID` split |
| M3 | parameter-efficiency gain | pruning vs never allocating / structured shrink | `C-TERM` + `C-SHRINK` |
| M4 | vs regularisation/rehearsal baselines; O(N) growth acknowledged | capacity and router quality | `C-TERM`; learned/reconstruction routing vs matched random routing |
| M5 | ACC/BWT after final merge | merge vs fewer experts; merge vs destructive removal | `C-RMERGE`, `B-EVICT`, `B-DENY` |
| M6 | near-largest-static performance with fewer parameters | saturation signal vs spawn rate | `C-RSPAWN`: realised spawn count with random timing |
| M7 | near-zero forgetting with one adapter per task | isolation vs router quality | `C-TERM`; `C-OID` to price GMM routing |
| M8 | competitive performance at fixed pool size | compression rule vs capacity | `C-SHRINK` at matched final capacity |

## Required reporting

Every run in either layer reports:

- retention, plasticity and forgetting **separately**;
- `param_total`, `param_active`, `param_peak`, `cold_bytes`, `storage_total`;
- `train_flops`, `infer_flops`, **`decision_flops`**, `consolidation_flops`, total;
- position on retention/plasticity versus capacity and versus compute frontiers;
- for merging methods: event-level merge loss, criterion/random-pair comparison, and
  recovery-after-merge with censoring.

The standardized panel additionally reports a shared frontier across methods. The
native-fidelity layer does **not** pretend scores from incompatible substrates are one
cross-method frontier.

## Questions the paper asks

**Q1 — attribution.** For each published gain, how much survives the relevant
terminal-capacity-, compute-, storage- and task-identity-matched controls in the native
setting?

**Q2 — mechanism under one substrate.** When allocation/routing/consolidation mechanisms are
placed on Long Sequence / MTL15 with one accounting protocol, which factors move the
retention–plasticity frontier?

**Q3 — which operation.** EXP-002 found a 0.204 simulator retention gap between merging and
LRU eviction at identical capacity, but no significant retention advantage over denying a
spawn. Do those distinctions survive in gradient-trained modular methods?

## Fidelity and exclusion rules

- Every native reproduction carries a fidelity note and, where possible, a check against a
  source-paper headline number with a predeclared tolerance.
- A method that does not reproduce is labelled **not reproduced** and excluded from claims
  about the source paper; its standardized mechanism implementation may still appear if it
  is independently valid.
- A standardized transplant is never described as reproducing the original paper.
- If controls show published gains survive, that is reported as a positive result for the
  field rather than reframed as failure of this project.

## Remaining blockers

The benchmark is no longer the conceptual blocker. The remaining blockers before Layer A
runs are:

1. exact standardized transplant definitions for M2/M3/M4/M6 and any later M5/M8 transplant;
2. model family, LoRA rank and shared routing substrate from `in-c0/plasticity-routing`.

Layer B can be prepared in parallel where a source implementation/native benchmark is
available, but no cross-substrate score may be used as if it were a common frontier.
