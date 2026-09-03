# Dependencies and boundaries

## Position in the CCS programme

This repository is one track of the Consolidating Cognitive Substrate programme. Sibling
repositories are independent; nothing here modifies their protocols, and their results are
not assumed here.

| Repository | Relationship |
| --- | --- |
| `in-c0/state-promotion` | Source of house conventions: budget matching, derived count-matched controls, decision-compute accounting, pilot/development/confirmatory separation. Its experimental protocol is not modified from here. A mechanically valid small-LM engineering pilot has now been run and reviewed; it was scientifically uninterpretable because the tested soft-prefix plastic representation underlearned. The track has predeclared an architecture-matched LoRA representation-sufficiency repair before further multi-arm runs. Those results are **not** evidence for this repository. |
| `in-c0/plasticity-routing` | **Blocking substrate dependency for Layer A and EXP-100.** Its synthetic EXP-001 exists and remains independent. Issue #1 had blocked any language-model EXP-002 until the first mechanically valid State Promotion engineering pilot was reviewed; that external prerequisite has now been met procedurally. However, `plasticity-routing` has **not** yet frozen the shared LM backbone, adapter family/rank, training budget, or routing substrate needed here. Therefore this repository's real-model standardized panel and EXP-100 remain blocked. See `experiments/EXP-100-PREREG-DRAFT.md` §11. |
| `in-c0/adaptive-commitment` | No dependency. Do not import its assumptions. No accessible repository is assumed here. |
| `in-c0/lifetime-integrity` | No dependency in either direction. Long-lifetime coherence is a separate failure class from capacity growth. |
| `in-c0/consolidating-cognitive-substrate` | Umbrella. Reconciles claims across tracks; does not own this protocol. |

## Dependency interpretation

The State Promotion pilot does **not** itself choose a substrate for this track. Its post-pilot
LoRA repair is informative engineering context only. In particular, do not copy its Qwen
rank/LR representation grid into the modular-consolidation methods panel as if it were a
`plasticity-routing` decision.

The blocking condition here is positive: wait until `plasticity-routing` explicitly freezes
or exports the common LM/adaptation/routing substrate required for the standardized panel.
The removal of its old upstream blocker is not equivalent to that substrate having been
selected.

Layer-B native-fidelity preparation is independent of this blocker where a source method
already defines its own model, adapter and routing substrate.
`experiments/M7-NATIVE-REANALYSIS-PREREG.md` is the first such design; it remains unrun and
has its own compute/provenance gate.

## Contamination rules

1. Results from this track are not evidence for any claim in another track, and the
   reverse holds.
2. This repository does not describe untested architecture as a result. Every claim in
   this repository is tagged with its evidence status: `conjecture`, `implemented`,
   `development-pilot`, `confirmatory`, `falsified`.
3. The CCS decomposition (ACCUMULATE → ALLOCATE → COMMIT) is a *conjecture*. This track
   investigates modular allocation/consolidation attribution. It does not assume the
   decomposition is correct, and nothing here should be cited as support for it.
4. Development seeds, calibration decisions and toy findings never migrate into a
   confirmatory claim without being re-established on confirmatory seeds.
5. A sibling track's development-selected model, rank, learning rate or routing mechanism is
   not inherited here unless the dependency contract explicitly exports it.

## Current evidence status

| Claim | Status |
| --- | --- |
| Every individual module operation already exists in the literature | `confirmed by audit` |
| The six-factor confound (routing/capacity/compute/task-ID/spawn/consolidation) is unresolved in prior work | `audit-supported`, absence claims re-checkable via `docs/LITERATURE-AUDIT-2026-09-02.md` §7 |
| Learned routing beats random routing at matched capacity | `development-pilot` |
| A bank with a bad router is worse than no bank | `development-pilot` |
| Consolidation beats a capacity-matched fixed bank (unbounded regime) | `falsified`, and explained: EXP-001 shows it cannot hold |
| Retention vs capacity is monotone under competent routing | `development-pilot`, replicated across 3 mechanisms |
| Consolidation in the unbounded regime is a compression claim, not a forgetting claim | `development-pilot` (follows from the above) |
| Under a binding ceiling, merging beats eviction | `development-pilot`, +0.204 retention, CI excludes zero |
| Under a binding ceiling, merging beats refusing to spawn | `regime-dependent`: no at `K*=6` (EXP-002/EXP-003); yes at `K*=12` and `K*=24` for ratios 1/3-5/6, dominating deny on both primary axes (EXP-003) |
| Consolidation adds a nondominated retention-plasticity operating point vs {deny, evict} | `development-pilot`, `SIMULATOR FRONTIER CONDITION: SATISFIED` (D7/D8), contiguous and replicated at two absolute `K*` |
| Useful consolidation needs capacity pressure **and** candidate diversity | `development-pilot` (EXP-003 mechanism); merge criterion collapses onto random at ceiling 2 |
| Merge-decision quality and merge cost are weakly coupled | **scoped** to the small-`K*`, tight-ceiling regime; reversed at `K*>=12`, ratio >=1/3 (EXP-003) |
| Real-model reproduction of the frontier effect | `NOT YET TESTED / BLOCKED` on the `plasticity-routing` substrate export |
| Architecture-paper gate | `CLOSED` (D9) |
| The merge criterion selects correct pairs | `development-pilot`; precision 0.45-0.81 given candidate slack, 0.07-0.19 when forced at ceiling 2 (EXP-003) |
| Any statement about real models from this repository | `no evidence` |

## No CI

This repository intentionally has no GitHub Actions. Gates run locally via `make test`.
