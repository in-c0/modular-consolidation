# Methods-paper experiment matrix

> **Layer B is now maintained in [`NATIVE-FIDELITY-LEDGER.md`](NATIVE-FIDELITY-LEDGER.md)**
> (2026-09-03). That ledger is authoritative for native source chains, frozen configurations,
> published targets, control classifications, blockers, compute estimates and per-method
> status. This matrix remains the selection rationale and the Layer-A view; where the two
> disagree about a native detail, the ledger wins.


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
| M6 | NORACL (arXiv:2604.27031) | **saturation-triggered neuron growth** | on saturation | none | — | no |
| M7 | Latent-LoRA (arXiv:2607.23837) | one adapter per task, **gradient-free routing** | per task | none | GMM over embeddings | no |
| M8 | FLAME (arXiv:2605.09355) | **fixed-size pool**, compression instead of expansion | none | low-rank compression | modality routers | no |

M1 exists because the lattice needs a reference isolating routing without allocation or
consolidation; no selected published method occupies that cell cleanly.

## D6: two evidence layers, not one forced benchmark

The previous requirement for one native-fidelity benchmark across all eight methods is
withdrawn. Cross-method mechanism comparability and attribution of a source paper's result
are different scientific objectives.

### Layer A — standardized mechanism panel

**Primary benchmark: Long Sequence / MTL15.** Use the two established task-agnostic Long
Sequence orders (Orders 3 and 4 as enumerated in `experiments/STANDARDIZED-TRANSPLANTS.md`),
task-agnostic inference, and task identity only for scoring and `C-OID`. All tasks are
classification tasks with accuracy as the common metric, making one retention–plasticity
frontier interpretable. The canonical sequence is not modified.

A method moved onto this substrate is a **standardized implementation of its mechanism**,
not a reproduction of the source paper's headline result. Admission requires the transplant
contract in `experiments/STANDARDIZED-TRANSPLANTS.md`; similarity by name is insufficient.

Current standardized-panel admission state after source verification:

| # | Standardized mechanism | Layer-A state | Why |
| --- | --- | --- | --- |
| M1 | fixed adapter bank + learned task-free router | **candidate** | constructed reference; shared model/rank/router still gated |
| M2 | `BEXP-LoRA`: one fresh adapter at each training task boundary | **candidate** | explicit P2 expansion reference, not DEN reproduction |
| M3 | MoCL-P composition + pruning transplanted to common PEFT | **blocked** | exact pruning-stat aggregation and task-agnostic adaptation unresolved |
| M4 | reconstruction-based novelty allocation | **blocked** | paper/code disagree on threshold, bottleneck and commitment implementation; reconcile before execution |
| M5 | manifold-aware merge rule | **excluded for now** | native CLIP-ViT geometry; LoRA transplant is semantic until validated |
| M6 | NORACL | **excluded for now** | native operation is layer-local neuron neurogenesis, not LoRA-expert spawn |
| M7 | Latent-LoRA/GMM routing | **candidate** | native Long Sequence/task-agnostic router is compatible; shared adapter family still gated |
| M8 | fixed-pool low-rank compression | **excluded for now** | native multimodal MoE substrate; LoRA transplant is semantic until validated |

The clean starting core is therefore M1 + M2 + M7 after the shared substrate is resolved,
not six methods forced into one architecture. M3/M4 join only when their listed blockers are
closed. M5/M6/M8 remain native-fidelity evidence unless independently validated Layer-A
transplants are added by dated amendment.

A recurrent-return Long Sequence variant is permitted only as a separately named,
predeclared stress test if returning-task/reinstatement behaviour must be exercised.

### Layer B — native-fidelity re-analysis

Each published method is reconstructed on its original substrate/benchmark as closely as
feasible and then receives the missing attribution controls. Only this layer licenses claims
about whether a published gain survives the protocol. MADE-IT, NORACL and FLAME remain here
unless and until a standardized transplant is independently validated; native MoCL-P and
Latent-LoRA also appear here for source-paper attribution.

## What the lattice adds to each

| # | Reported comparison (as published) | Predicted confound the lattice tests | Missing control(s) |
| --- | --- | --- | --- |
| M1 | — | — | reference point |
| M2 | vs fixed-capacity baselines of different size | growth vs routing/capacity | `C-TERM`; explicit task-free vs `C-OID` split |
| M3 | parameter-efficiency gain | pruning vs never allocating / structured shrink | no-prune counterpart, `C-TERM`, `C-SHRINK`, `C-OID` |
| M4 | vs regularisation/rehearsal baselines; O(N) growth acknowledged | capacity and novelty/router quality | paper-contract vs code-fidelity reconciliation first; then `C-TERM`, matched random routing, `C-RSPAWN` |
| M5 | ACC/BWT after final merge | merge vs fewer experts; merge vs destructive removal | `C-RMERGE`, `B-EVICT`, `B-DENY` |
| M6 | near-largest-static performance with fewer parameters | native saturation signal/growth vs capacity and event timing | native matched-capacity control + `C-RSPAWN`-style timing control where valid |
| M7 | near-zero forgetting with one adapter per task | isolation/adapter geometry vs router quality | matched random router, `C-OID`, terminal-capacity control |
| M8 | competitive performance at fixed pool size | compression rule vs capacity | `C-SHRINK` at matched final capacity |

## Required reporting

Every run in either layer reports:

- retention, plasticity and forgetting **separately**;
- `param_total`, `param_active`, `param_peak`, `cold_bytes`, `storage_total`;
- `train_flops`, `infer_flops`, **`decision_flops`**, `consolidation_flops`, total;
- position on retention/plasticity versus capacity and versus compute frontiers;
- for merging methods: event-level merge loss, criterion/random-pair comparison, and
  recovery-after-merge with censoring.

The standardized panel additionally reports a shared frontier across admitted mechanisms.
The native-fidelity layer does **not** pretend scores from incompatible substrates are one
cross-method frontier.

## Questions the paper asks

**Q1 — attribution.** For each published gain, how much survives the relevant
terminal-capacity-, compute-, storage- and task-identity-matched controls in the native
setting?

**Q2 — mechanism under one substrate.** When scientifically portable mechanisms are placed
on Long Sequence / MTL15 with one accounting protocol, which factors move the
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
- A semantic P3 transplant cannot enter Layer A under the source method's name merely to
  make the panel look complete.
- If paper prose and public source code disagree on an executable mechanism, the conflict is
  recorded and resolved by a dated pre-run contract; outcome scores may not choose the
  interpretation.
- If controls show published gains survive, that is reported as a positive result for the
  field rather than reframed as failure of this project.

## Remaining blockers

The benchmark is no longer the conceptual blocker. Before any Layer-A run:

1. `in-c0/plasticity-routing` must fix the shared model family, adapter/rank and routing
   substrate;
2. M3 must resolve the authoritative aggregation used for its pruning statistic and the
   task-agnostic head/routing adaptation;
3. M4 must reconcile the paper's dynamic novelty/commitment contract with the inconsistent
   fixed-threshold public simulation scripts before freezing any implementation.

M1/M2/M7 can be implemented once blocker 1 lifts. Layer B preparation can continue in
parallel where a source implementation/native benchmark is available, but no cross-substrate
score may be used as if it were a common frontier.
