# Dependencies and boundaries

## Position in the CCS programme

This repository is one track of the Consolidating Cognitive Substrate programme. Sibling
repositories are independent; nothing here modifies their protocols, and their results are
not assumed here.

| Repository | Relationship |
| --- | --- |
| `in-c0/state-promotion` | Source of house conventions: budget matching, derived count-matched controls, decision-compute accounting, pilot/development/confirmatory separation. Its experimental protocol is not modified from here. |
| `in-c0/plasticity-routing` | **Blocking dependency.** EXP-100 cannot be frozen or executed until its design lessons are available. See `experiments/EXP-100-PREREG-DRAFT.md` §11. Not yet created as of 2026-09-02. |
| `in-c0/adaptive-commitment` | No dependency. Do not import its assumptions. Not yet created as of 2026-09-02. |
| `in-c0/lifetime-integrity` | No dependency in either direction. Long-lifetime coherence is a separate failure class from capacity growth. |
| `in-c0/consolidating-cognitive-substrate` | Umbrella. Reconciles claims across tracks; does not own this protocol. |

## Contamination rules

1. Results from this track are not evidence for any claim in another track, and the
   reverse holds.
2. This repository does not describe untested architecture as a result. Every claim in
   this repository is tagged with its evidence status: `conjecture`, `implemented`,
   `development-pilot`, `confirmatory`, `falsified`.
3. The CCS decomposition (ACCUMULATE → ALLOCATE → COMMIT) is a *conjecture*. This track
   investigates the ALLOCATE stage. It does not assume the decomposition is correct, and
   nothing here should be cited as support for it.
4. Development seeds, calibration decisions and toy findings never migrate into a
   confirmatory claim without being re-established on confirmatory seeds.

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
| Under a binding ceiling, merging beats refusing to spawn | `unresolved`, retention null concealing a significant plasticity/forgetting trade |
| The merge criterion selects correct pairs | `development-pilot` (opportunistic merges only; degrades to ~chance when merges are forced) |
| Merge-decision quality and merge cost are weakly coupled | `development-pilot`, replicated in two regimes |
| Any statement about real models | `no evidence` |

## No CI

This repository intentionally has no GitHub Actions. Gates run locally via `make test`.
