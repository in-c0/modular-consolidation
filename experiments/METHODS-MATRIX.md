# Methods-paper experiment matrix

Per owner decision D3 (`docs/OWNER-DECISIONS.md`), the primary near-term paper is a
methods/evaluation paper about **attribution** in modular continual learning. It does not
propose a new policy. It re-analyses representative published methods under the control
lattice and asks, for each, which factor its reported gain is actually attributable to.

**Status: design. No re-analysis has been run.** Nothing below is a claim about any of these
papers; the "predicted confound" column states what the lattice would test, not what is
true.

## Selection rule

Methods are selected to **span the factor space**, not by popularity or reported score. One
representative per cell, chosen for (a) reproducibility from the paper, (b) a clearly
identifiable allocation/consolidation mechanism, (c) coverage of a distinct factor.
Selection is fixed here before any re-analysis is run; additions require a dated amendment.

| # | Method | Factor it exemplifies | Allocation | Consolidation | Routing | Task IDs |
| --- | --- | --- | --- | --- | --- | --- |
| M1 | Fixed LoRA bank + learned router (constructed reference, not from a paper) | routing at fixed capacity | none | none | learned | no |
| M2 | Progressive Networks / DEN family | unbounded expansion | per task | none | task-indexed | yes |
| M3 | MoCL-P — *Learn it or Leave it* (Repl4NLP 2024, arXiv:2406.18708) | composition + **pruning** | per task | prune | composition weights | yes |
| M4 | Zero-Leakage Reconstruction Routing (arXiv:2604.14375) | **task-free discovery**, no consolidation at all | on novelty | none | reconstruction error | no |
| M5 | MADE-IT — Manifold-Aware Expert Evolution (arXiv:2604.22464) | **merge-based consolidation** | on novelty | merge | training-free, subspace | no |
| M6 | NORACL (arXiv:2604.27031) | **saturation-triggered growth** | on saturation | none | — | no |
| M7 | Latent-LoRA (arXiv:2607.23837) | one adapter per task, **gradient-free routing** | per task | none | GMM over embeddings | no |
| M8 | FLAME (arXiv:2605.09355) | **fixed-size pool**, compression instead of expansion | none | low-rank compression | modality routers | no |

M1 exists because the lattice needs a reference that isolates routing with no allocation or
consolidation at all, and no published method occupies that cell cleanly.

## What the lattice adds to each

The column that matters is the last one: the control the original paper did not run.

| # | Reported comparison (as published) | Predicted confound the lattice tests | Control the lattice adds |
| --- | --- | --- | --- |
| M1 | — | — | reference point |
| M2 | vs fixed-capacity baselines of different size | growth vs routing | `C-TERM`: fixed bank at M2's realised final size |
| M3 | "up to 3× better parameter efficiency" | pruning vs never having allocated | `C-TERM` + `C-SHRINK`: does compressing an unpruned bank to the same size match it? |
| M4 | vs regularisation/rehearsal baselines; O(N) growth acknowledged | capacity, entirely | `C-TERM`; and whether reconstruction routing beats random routing at matched capacity |
| M5 | ACC and BWT after the final merge | merge vs having fewer experts; merge vs **evict** | `C-RMERGE` (merge-count-matched random pairs), `B-EVICT` (destroy instead of pool), `B-DENY` |
| M6 | "on par with the largest static baseline, 10–22% fewer parameters" | saturation signal vs spawn *rate* | `C-RSPAWN`: same spawn count, random timing |
| M7 | near-zero forgetting, one adapter per task | parameter isolation, not routing | `C-TERM`; oracle-ID upper bound `C-OID` to price the GMM router |
| M8 | competitive at fixed pool size | compression vs capacity | `C-SHRINK` at matched final capacity |

## Required reporting for every re-analysed method

Each method is re-run (or reconstructed) and reported on the same axes:

- retention, plasticity and forgetting **separately** — EXP-002 showed a null on retention
  concealing two significant effects in opposite directions;
- `param_total`, `param_active`, `param_peak`, `cold_bytes`, `storage_total`;
- `train_flops`, `infer_flops`, **`decision_flops`**, `consolidation_flops`, total;
- position on the retention-vs-capacity and retention-vs-compute frontiers;
- for any method that merges: per-event merge loss decomposed into decision and mechanism
  components, plus recovery.

## The two questions the paper asks of the field

**Q1 — attribution.** For each published gain, how much survives a terminal-capacity-matched
fixed bank with the same router and the same total compute?

**Q2 — which operation.** EXP-002 found a 0.204 retention gap between *merging* and
*evicting* at identical capacity, and no significant gap between merging and simply *not
spawning*. Papers routinely describe "pruning redundant experts" and "merging redundant
experts" interchangeably. On this evidence they are different operations with very different
consequences, and the paper asks each method which one it actually performs.

## Honest scope

- Reimplementation is imperfect; every method is reported with a fidelity note and, where
  available, a check that the reimplementation reproduces the paper's headline number within
  a stated tolerance. Where it does not, the method is reported as **not reproduced** and
  excluded from the attribution claims rather than quietly included.
- The paper does not claim any of these methods is wrong. It claims their reported gains
  have not been *attributed*, which is a statement about the field's evaluation protocol.
- If the controls show the gains survive, that is a positive result for the field and is
  reported as such.

## Open

- Benchmark for re-analysis: **[OPEN]**, gated on the same substrate decision as EXP-100
  (owner decision D1: small LM + LoRA). A shared benchmark across all eight methods is
  required for the frontier plots to be meaningful.
- Compute budget per method: **[OPEN]**.
