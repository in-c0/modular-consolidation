# EXP-003 — binding-capacity pressure phase diagram (preregistration)

**Status: PREDECLARED DEVELOPMENT SIMULATOR EXPERIMENT — UNRUN.**

This file is committed before the EXP-003 sweep runner or any EXP-003 result. It records the
complete grid, seeds, stream scaling, outcomes and interpretation rules required by owner
decision D5. It is not confirmatory evidence about real models and does not lift the EXP-100
execution gate.

## Question

EXP-002 found that at `K*=6, ceiling=3`, B-MERGE was much safer than destructive eviction
but did not beat B-DENY on retention. The obvious alternative explanation is that a small
number of recurring skills makes refusing novelty unusually safe.

EXP-003 asks:

> As binding memory pressure increases, and as the absolute number of latent skills grows,
> does merging remain only a stability–plasticity tradeoff relative to denying or evicting a
> spawn, or is there a reproducible region in which it improves the retention–plasticity
> frontier at identical capacity?

The experiment is a **phase diagram**, not a search for a winning ceiling.

## Frozen grid

Absolute skill count and capacity ratio are crossed:

- `K* ∈ {6, 12, 24}`
- `ceiling / K* ∈ {1/6, 1/3, 1/2, 2/3, 5/6}`

All ratios are exact integers for all three `K*` values:

| K* | 1/6 | 1/3 | 1/2 | 2/3 | 5/6 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 1 | 2 | 3 | 4 | 5 |
| 12 | 2 | 4 | 6 | 8 | 10 |
| 24 | 4 | 8 | 12 | 16 | 20 |

Every cell is reported. No cell may be dropped because its result is inconvenient or because
another ratio looks more interesting.

## Frozen arms

At every cell, run exactly:

- `B-DENY`
- `B-MERGE`
- `B-EVICT-LRU`
- `B-EVICT-RAND`
- `B-MERGE-RAND`

All arms have the same hard live-module ceiling in a cell. As in EXP-002, capacity is equal
by construction once the ceiling binds; the operation used to free or refuse a slot is the
factor of interest.

## Frozen stream scaling

EXP-002 used `K*=6`, `n_segments=18`, so EXP-003 preserves **three expected segments per
latent skill** by setting:

`n_segments = 3 * K*`

The stream probabilities remain unchanged from EXP-002:

- `recur_prob = 0.30`
- `near_dup_prob = 0.30`
- therefore the novel-attempt probability before `K*` is exhausted remains `0.40`
- `region_scale = 0.7`
- all other `StreamConfig` values remain at their committed defaults.

This fixes expected exposure density and recurrence statistics as `K*` grows while allowing
realised stochastic counts to vary naturally. **Do not resample a seed to force a desired
skill count or recurrence mix.** The runner records, per stream, realised kind counts,
distinct skills observed and per-skill exposure dispersion. A stream that does not introduce
all `K*` skills is reported as such rather than silently replaced.

No recurrence parameter may be changed after seeing this phase diagram. A recurrence sweep,
if motivated, is a new separately predeclared experiment.

## Frozen seeds

Use eight paired **development** seeds:

`900, 901, 902, 903, 904, 905, 906, 907`

These are development/simulator seeds under the repository seed discipline. EXP-002's
historical seeds `0–7` remain untouched; this experiment does not rewrite or relabel that
result.

Within each `(K*, seed)`, the same generated stream is reused across all ceiling ratios and
all five slot policies. The seed is therefore paired across both policy and pressure.

## Pressure diagnostic: K6 is reported, not used to select cells

For every cell report the A3 learned-router retention difference between unbounded capacity
and that ceiling, along with the existing K6 threshold (`0.05`). `ceiling < K*` is true by
construction throughout the grid.

Unlike EXP-002 admissibility, **K6b is not an inclusion rule for EXP-003**: high-ceiling cells
are intentionally present to show the low-pressure end of the phase diagram. A cell below
the historical K6 cost threshold remains in every plot and table and is labelled as such.

## Primary outcomes

The primary comparison is paired `B-MERGE − B-DENY`, reported **separately** for:

1. retention;
2. plasticity.

For each cell report the paired mean difference and paired-bootstrap 95% CI. Do not combine
retention and plasticity into a scalar score.

Also report paired `B-MERGE − B-EVICT-LRU` for retention and plasticity over the same complete
grid. B-EVICT-RAND and B-MERGE-RAND remain required criterion controls.

## Explanatory outcomes

Report, without promoting them to primary endpoints:

- forgetting;
- merge precision;
- per-event total, decision and mechanism merge loss;
- recovery-after-merge and recovery censoring;
- live/peak parameters, storage and routing decision compute;
- realised spawn/merge/evict counts;
- stream kind counts, distinct skills observed and exposure dispersion.

## Interpretation fixed before the run

The owner decision is binding:

1. **Operating-point result.** If merge only exchanges retention for plasticity relative to
   deny across the grid, while remaining safer than eviction, it stays in the methods paper
   as a stability–plasticity operating-point result.
2. **Architecture signal.** Re-open the architecture-paper question only if the phase diagram
   contains a **reproducible region**, across adjacent pressure values and/or replicated at a
   second absolute `K*`, in which B-MERGE improves the retention–plasticity frontier relative
   to deny and evict at identical capacity. A single favourable cell is insufficient.
3. **Isolated-cell rule.** Any advantage appearing at only one ceiling ratio is exploratory
   until reproduced at a second absolute `K*`.
4. **No rescue tuning.** Recurrence, stream difficulty, merge threshold, routing or seed set
   may not be altered in response to the curves.

A mean-level tradeoff (one primary dimension up, the other down) is **not** called frontier
expansion. Statistical uncertainty for both primary dimensions is shown so any regime-level
claim can be judged from the complete pattern rather than a post-hoc scalarisation.

## Outputs

The intended runner writes one machine-readable payload containing:

- the exact frozen design constants;
- stream diagnostics for every `(K*, seed)`;
- every arm/seed/cell row;
- K6 pressure diagnostics for every cell;
- paired retention and plasticity comparisons for MERGE−DENY and MERGE−EVICT-LRU.

A human-readable result document may be written **only after** that payload exists. It must
report all 15 grid cells.

## Non-claims

- This remains a synthetic closed-form development simulator.
- It does not establish an architecture result, even if a regime appears.
- It does not lift EXP-100's dependency on `in-c0/plasticity-routing`.
- It does not license a recurrence sweep, benchmark modification or real-model run without
  the separately required gates.
