# The arm lattice — one factor at a time

## Why not the naive ladder

The track brief proposed a candidate hierarchy:

1. single fixed adapter → 2. fixed bank → 3. learned routing → 4. dynamic spawning →
5. spawn + merge/compress → 6. spawn + merge + retirement/reuse

The brief also said not to implement it blindly. It should not be implemented as written, because
**each rung changes more than one factor at a time.** Going from rung 2 to rung 4 changes capacity,
allocation policy, routing distribution, and decision compute simultaneously. A win at rung 4 would
be uninterpretable — which is exactly the confound identified in the literature audit §3.

The ladder is therefore replaced by a **lattice** in which each edge changes exactly one factor,
plus a set of derived controls whose configuration is computed *from the realised behaviour* of the
arm they are controlling for.

## Factors

| Symbol | Factor | Levels |
| --- | --- | --- |
| F1 | routing | none / random-fixed / learned / oracle-task-ID |
| F2 | capacity | 1 module / fixed-K / dynamic (realised N) |
| F3 | compute | base / augmented-to-match |
| F4 | task identity | task-free / boundaries given / IDs given |
| F5 | allocation | fixed / spawn-by-criterion / spawn-random-timing |
| F6 | consolidation | none / merge-by-criterion / merge-random-pairs / merge+retire+reinstate |
| F7 | slot policy under a binding ceiling | deny / evict-LRU / evict-random / merge-best / merge-random |
| F8 | compression | none / structured width reduction to a capacity target |

`F1` gained a `soft` level (density-gated mixture over all live modules) after EXP-001, and
`F7`/`F8` were added after Amendment B. `F7` is the factor that matters most: it is the only
place where capacity is held equal *by construction* rather than by matching.

## Primary arms

| ID | Name | F1 | F2 | F3 | F4 | F5 | F6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `A0` | frozen backbone, no adaptation | none | 0 | base | free | fixed | none |
| `A1` | single adapter, sequential | none | 1 | base | free | fixed | none |
| `A2` | fixed bank, random fixed routing | random | K | base | free | fixed | none |
| `A3` | fixed bank, learned routing | learned | K | base | free | fixed | none |
| `A4` | dynamic spawn, learned routing | learned | dyn | base | free | criterion | none |
| `A5` | spawn + merge | learned | dyn | base | free | criterion | merge |
| `A6` | spawn + merge + retire/reinstate | learned | dyn | base | free | criterion | full |

`A0..A6` is the ladder. On its own it proves nothing. The controls below are the experiment.

## Derived controls — configured from realised behaviour

These cannot be specified in advance; each is generated after its target arm runs, in the same way
State Promotion derives its count-matched random control from the promotion arm's realised write
count. Their configuration is a deterministic function of the target run's manifest.

| ID | Controls for | Construction |
| --- | --- | --- |
| `C-TERM(A4)` | capacity, not allocation | fixed bank with `K = k_final(A4)`, learned routing, same stream, same seed |
| `C-TERM(A6)` | capacity, not consolidation | fixed bank with `K = k_final(A6)` |
| `C-PEAK(A6)` | peak capacity | fixed bank with `K = k_peak(A6)` |
| `C-FLOP(A3←A6)` | compute | `A3` trained with additional steps until `total_algorithmic_flops` matches `A6` |
| `C-RMERGE(A5)` | merge criterion vs merge rate | identical to `A5` but merged pairs chosen uniformly at random, merge count and timing copied from `A5`'s manifest |
| `C-RSPAWN(A4)` | spawn criterion vs spawn rate | spawn count copied from `A4`, timings drawn uniformly over the stream |
| `C-OID(A3)` | value of task inference | `A3` with oracle task IDs replacing the learned router (upper bound on routing) |
| `C-BOUND(A4)` | value of boundary knowledge | `A4` with true segment boundaries supplied to the spawn trigger |

## Binding-ceiling arms (added 2026-09-02, Amendment B)

Every arm holds the same ceiling and therefore the same live-module count, `param_total` and
`storage_total` at every step. They differ only in how a slot is freed when a spawn is
wanted and the bank is full. This is the only part of the lattice where capacity needs no
matching at all, because it cannot differ.

| ID | On full | Isolates |
| --- | --- | --- |
| `B-DENY` | refuse to spawn | the baseline the literature almost never reports |
| `B-EVICT-LRU` | delete least-recently-used, then spawn | destroying knowledge to make room |
| `B-EVICT-RAND` | delete a random module, then spawn | the eviction *criterion*, vs the eviction rate |
| `B-MERGE` | pool the most similar pair, then spawn | pooling knowledge to make room |
| `B-MERGE-RAND` | pool a random pair, then spawn | the merge *criterion*, vs merging at all |

`C-SHRINK(A)` was also added: structured width reduction of an unconsolidated bank down to
a target arm's realised `param_total`. It reduces capacity **without combining knowledge**,
so if it matches a merging arm then "consolidation" collapses into the capacity factor.

## What each comparison licenses

| Comparison | Licensed conclusion if significant |
| --- | --- |
| `A3` vs `A2` | learned routing beats random routing at identical capacity → **routing** has value |
| `A3` vs `A1` | a bank beats one adapter → **capacity/isolation** has value |
| `C-OID(A3)` vs `A3` | gap = the price of not knowing task identity |
| `A4` vs `C-TERM(A4)` | **dynamic allocation** has value beyond ending at N modules |
| `A4` vs `C-RSPAWN(A4)` | the **spawn criterion** has value beyond the spawn rate |
| `C-BOUND(A4)` vs `A4` | gap = the price of task-free operation |
| `A5` vs `A4` at matched `param_total` | **merging** has value; if `A5` merely shrinks the model at equal accuracy that is still a win, reported on the frontier |
| `A5` vs `C-RMERGE(A5)` | the **merge criterion** has value beyond merging at all |
| `A6` vs `A5` | **retirement/reinstatement** has value once cold storage is charged |
| `A6` vs `C-TERM(A6)`, `C-PEAK(A6)` | the whole lifecycle beats a fixed bank of the same size |
| `A6` vs `C-FLOP(A3←A6)` | the lifecycle beats simply spending the same compute on a fixed bank |

| `B-MERGE` vs `B-EVICT-LRU` | **pooling** beats destroying at identical capacity |
| `B-MERGE` vs `B-DENY` | consolidation is worth admitting new modules at all |
| `B-MERGE` vs `B-MERGE-RAND` | the merge criterion contributes beyond merging |
| `B-EVICT-LRU` vs `B-EVICT-RAND` | the eviction criterion contributes beyond evicting |
| `A5` vs `C-SHRINK(A5)` | merging beats plain compression to the same size |

## The result that would falsify the track

If `C-TERM(A6)` matches or beats `A6` on the retention–plasticity frontier at equal `param_total`
and equal `total_algorithmic_flops`, then **dynamic allocation and consolidation contribute nothing
that choosing the right fixed size would not have contributed.** That is a publishable negative
result and must be reported as the headline, not buried.

**This already happened.** EXP-000 found exactly that, and EXP-001 then explained why it was
inevitable in the unbounded regime: the retention-versus-capacity curve is monotone, so there
is nothing for consolidation to fix. The falsification is recorded, and the track moved to the
binding-ceiling regime where the comparison is not degenerate.

In the binding-ceiling regime the falsifying result is different: if `B-MERGE` matches
`B-EVICT-LRU`, then pooling knowledge is worth nothing over destroying it, and consolidation
has no defensible claim at all. EXP-002 did not find that.
