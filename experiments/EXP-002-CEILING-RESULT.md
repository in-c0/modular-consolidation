# EXP-002 — the binding-ceiling regime

**Status: DEVELOPMENT SIMULATOR.** Not evidence about real models. It is reported because
it settles the shape of the track's primary hypothesis and produces one significant
positive, one significant negative, and one clean null.

- Code: `scripts/run_ceiling.py`
- Stream: CAMS, `K* = 6`, 18 segments, `region_scale = 0.7`
- Ceiling: 3 live modules (half of `K*`)
- Seeds: 0–7 (development), paired across arms
- Learner: per-module ridge over frozen random features

## Why this experiment exists

EXP-001 (`docs/BENCHMARK-POLICY.md` Amendment B) established that in the **unbounded**
regime the retention-versus-capacity curve is monotone non-decreasing under any competent
routing: a module that is never selected cannot damage a prediction, so spare modules cost
parameters but not accuracy. Consolidation therefore cannot improve retention at matched
capacity, and the original H1 was malformed.

Consolidation can only be irreducible to capacity when capacity cannot simply be increased.
Under a ceiling below `K*`, a policy that wants a new module must free a slot, and **how**
it frees that slot is a decision capacity accounting cannot make for it.

## Design

Every arm holds `cap = 3`. All five end each step at the **same live-module count, the same
`param_total` (576) and the same `storage_total` (33 408)** — verified in the results table.
Capacity is equal by construction, so any difference is attributable to the slot decision.

| arm | on-full behaviour |
| --- | --- |
| `B-DENY` | refuse to spawn; keep using an existing module |
| `B-EVICT-LRU` | delete the least recently used module, then spawn |
| `B-EVICT-RAND` | delete a uniformly random module, then spawn — eviction criterion control |
| `B-MERGE` | pool the two most functionally similar modules, then spawn |
| `B-MERGE-RAND` | pool two uniformly random modules, then spawn — merge criterion control |

## Admissibility (K6, predeclared in Amendment B)

- K6a `ceiling (3) < K* (6)` — **pass**
- K6b ceiling cost `0.097 >= 0.05` — **pass** (unbounded 0.848 vs ceiling 0.751)

## Results, seeds 0–7

| arm | retention | plasticity | forgetting | P_total | storage | spawns | merges | evictions | merge loss | recovery | merge precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B-DENY | **0.762** | 0.775 | **0.032** | 576 | 33 408 | 3.0 | 0 | 0 | — | — | — |
| B-EVICT-LRU | 0.541 | **0.854** | 0.324 | 576 | 33 408 | 19.0 | 0 | 16.0 | — | — | — |
| B-EVICT-RAND | 0.500 | 0.854 | 0.365 | 576 | 33 408 | 16.6 | 0 | 13.6 | — | — | — |
| B-MERGE | 0.745 | 0.798 | 0.071 | 576 | 33 408 | 7.9 | 4.9 | 0 | 0.027 | 0.74 | 0.50 |
| B-MERGE-RAND | 0.736 | 0.805 | 0.086 | 576 | 33 408 | 8.8 | 5.8 | 0 | 0.042 | 0.71 | 0.36 |

Paired bootstrap 95% CIs (`*` = excludes zero):

| comparison | retention | plasticity | forgetting |
| --- | --- | --- | --- |
| B-MERGE − B-DENY | −0.017 [−0.042, +0.007] | **+0.023 [+0.000, +0.050] \*** | **+0.039 [+0.021, +0.060] \*** |
| B-MERGE − B-EVICT-LRU | **+0.204 [+0.125, +0.296] \*** | — | — |
| B-MERGE − B-MERGE-RAND | +0.009 [−0.024, +0.039] | −0.007 [−0.018, +0.005] | — |
| B-EVICT-LRU − B-EVICT-RAND | +0.042 [−0.000, +0.105] | — | — |
| B-EVICT-LRU − B-DENY | **−0.221 [−0.301, −0.155] \*** | **+0.079 [+0.060, +0.102] \*** | **+0.292 [+0.226, +0.370] \*** |

## Finding 1 — consolidation preserves what eviction destroys (significant)

At identical capacity, merging retains **0.204 more** than evicting. Eviction is
catastrophic: it buys the highest plasticity in the study (0.854) and pays 0.292 additional
forgetting for it. Merging recovers essentially all of that loss.

This is the clearest positive result the track has produced, and it is the defensible form
of the consolidation claim: **merging is a safe way to admit new modules under a binding
ceiling.** Note what it is *not* — it is not evidence that merging beats a static bank.

## Finding 2 — merging does not beat refusing to spawn (null, and it matters)

`B-MERGE` versus `B-DENY` on retention: −0.017, CI spans zero. Decomposed, the two arms are
making a **pure stability–plasticity trade**, both halves significant:

- plasticity **+0.023**, CI excludes zero — merging admits new skills that denial cannot;
- forgetting **+0.039**, CI excludes zero — and pays for them.

Net retention is a wash. So the honest statement is not "consolidation helps" nor
"consolidation is useless", but: **consolidation converts stability into plasticity at
roughly par.** Whether that is a good trade depends on the deployment, which is a claim
about the frontier, not about a single accuracy number — exactly as Amendment B predicted.

Anyone reporting that merging reduces forgetting should be asked which arm they compared
against. Against eviction it plainly does. Against simply not spawning, in this setting, it
plainly does not.

## Finding 3 — the merge criterion has no detectable value under forced consolidation (null)

`B-MERGE` versus `B-MERGE-RAND` on retention: +0.009, CI spans zero. Yet the criterion is
demonstrably doing something:

- ground-truth merge precision 0.50 versus 0.36;
- mean per-event merge loss 0.027 versus 0.042.

The criterion picks better pairs and each of its merges is less damaging, and none of that
survives to the aggregate. This reproduces EXP-000 Finding 4 in a completely different
regime and strengthens it: **merge-decision quality and merge outcome are only weakly
coupled.**

Note the precision collapse against EXP-000, where the same criterion scored 1.00. There,
merges fired only when a good pair existed. Here they are **forced** by the need for a
slot, so the criterion must pick the best available pair even when no pair is good.
**Forced consolidation degrades merge-decision quality** — a mechanism worth stating
explicitly, because deployed systems under memory pressure are in the forced regime, not
the opportunistic one.

## Finding 4 — most merge damage is recovered

Mean recovery 0.74 of the merge loss within 16 chunks, and per-event loss is small
(0.027–0.042) relative to the retention differences between arms. Merging is not
principally damaging *at the moment it happens*; what matters is the knowledge it fails to
keep separate afterwards.

## Consequences for the protocol

1. **H1 is replaced.** The primary hypothesis is no longer "consolidation beats a
   capacity-matched fixed bank on retention". It is a binding-ceiling, frontier-shaped
   claim; see `experiments/EXP-100-PREREG-DRAFT.md` §3.
2. **`B-DENY` becomes a mandatory baseline.** It is trivial, it is almost never reported in
   the modular CL literature, and here it is the strongest arm on retention.
3. **Eviction must be reported separately from merging.** Papers that describe "pruning
   redundant experts" are, on this evidence, describing a very different operation from
   merging them, with a 0.20 retention gap between the two.
4. **Report the stability–plasticity decomposition, never retention alone.** A null on
   retention concealed two significant effects in opposite directions.

## Threats to validity

- Single toy learner (closed-form ridge). Merge is near-lossless here; in gradient-trained
  adapters the merge operator is the dominant damage source in the literature, which would
  move Finding 1 and possibly Finding 3.
- `B-DENY` benefits from `K*` being small and the stream revisiting skills. With many more
  distinct skills than slots, denial should degrade faster than merging; the ceiling/`K*`
  ratio is a sweep this experiment has not run.
- LRU is one eviction rule among many; a better rule might close part of the 0.204 gap.
- Recovery is measured over 16 chunks on a probe frozen at merge time; longer horizons are
  censored and reported as such.
