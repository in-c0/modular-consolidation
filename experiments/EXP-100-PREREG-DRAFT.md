# EXP-100 preregistration — DRAFT, NOT FROZEN

> **Execution gate.** This protocol must not be frozen or executed as a confirmatory
> experiment until the design lessons from `in-c0/plasticity-routing` are available and
> recorded in §11. Sections marked **[OPEN]** are deliberately unresolved and depend on
> that input. Running EXP-100 before the gate lifts produces an engineering pilot at best
> and an uninterpretable result at worst.

Version: draft-0.1 (2026-09-02). Any change after freezing requires a numbered amendment
in this file with a date and the reason, following the State Promotion convention.

---

## 1. Question

> In a task-free continual stream, when total stored parameters, active parameters per
> input, cold storage bytes, replay bytes and total algorithmic compute (including
> routing decision compute) are all matched, does a policy that **spawns, merges, retires
> and reinstates** modules achieve a better retention–plasticity frontier than a fixed
> bank of the same terminal size with the same learned routing?

Equivalently, and more bluntly: **does consolidation buy anything that capacity cannot?**

The development pilot (EXP-000) answered *no* on a toy stream. EXP-100 asks whether that
survives a real model, a real adapter, and a stream with genuine interference.

## 2. Why this is the narrow question

The literature audit (`docs/LITERATURE-AUDIT-2026-09-02.md`) establishes that every
individual operation — spawning, task-free discovery, merging, pruning, routing — already
exists. What does not exist is a matched-budget decomposition that says which of them is
responsible for observed gains. The contribution claimed here is that decomposition and
its controls, not an architecture.

## 3. Hypotheses

**H1 (primary, consolidation).** At matched `param_total`, `param_active`,
`storage_total` and `total_algorithmic_flops`, arm `A6` achieves higher retention than
`C-TERM(A6)` with a paired-bootstrap 95% CI excluding zero, while retaining at least 95%
of `C-TERM(A6)`'s plasticity.

**H2 (allocation).** `A4` achieves higher retention than `C-RSPAWN(A4)` (spawn-count- and
rate-matched, random timing), establishing that the spawn *criterion* contributes beyond
the spawn *rate*.

**H3 (merge decision).** `A5` achieves lower total merge loss per event than
`C-RMERGE(A5)` (merge-count- and time-matched, random pairs), establishing that the merge
*criterion* contributes beyond merging at all.

**H4 (frontier).** `A6` lies on the Pareto front of retention against `param_total` and of
retention against `total_algorithmic_flops`, jointly with all controls.

H1 is the hypothesis the paper stands or falls on. H2–H4 remain interpretable if H1 fails.

## 4. Falsification

The track is falsified — and the resulting paper is a negative-result paper, published as
such — if any of the following hold on confirmatory seeds:

- `C-TERM(A6)` matches or beats `A6` on retention at equal capacity and compute (CI
  includes zero or favours the control);
- `A6` is off the Pareto front in §3 H4;
- `A5` and `C-RMERGE(A5)` are indistinguishable, i.e. the merge criterion is inert;
- gains disappear once `decision_flops` are included in the compute match.

EXP-000 already produced the first of these on a toy. That is stated in the paper
regardless of what EXP-100 finds.

## 5. Arms

`A1`–`A6` and the derived controls exactly as specified in `docs/ARMS.md`. Every edge in
the primary ladder changes one factor, enforced by
`tests/test_policies.py::test_adjacent_primary_arms_differ_in_exactly_one_factor`.

Derived controls are configured from the *realised manifest* of the arm they control for,
computed by `policies.derive_controls`, so their configuration cannot be selected after
seeing which comparison is favourable.

## 6. Budget matching

Matched across all arms:

- base model and frozen backbone (identical checkpoint and revision);
- `param_total` at end of lifetime, matched to within 2% for capacity-matched controls;
- `param_active` mean, reported and matched where the arm permits;
- online examples/tokens seen;
- replay bytes (zero for all arms in EXP-100 unless §11 changes it);
- cold-storage bytes charged to `storage_total`;
- `total_algorithmic_flops`, with `decision_flops` reported separately **and** included.

Where an arm cannot be matched on a dimension, the mismatch is reported as a number in the
results table, never described as "approximately matched".

## 7. Benchmark

CAMS-v0 for development. **[OPEN]** The confirmatory stream is not yet chosen. It must:

- contain recurrence, near-duplicates and genuinely distinct skills, with a known `K*`;
- create genuine interference, not merely distinct input regions — EXP-000 showed a stream
  with separated regions is solved by capacity alone and cannot discriminate;
- pass the method-independent admissibility criterion in `scripts/calibrate_stream.py`
  (single-adapter to oracle-ID headroom ≥ 0.15, no ceiling), calibrated on development
  seeds disjoint from confirmatory seeds;
- be fixed before any confirmatory seed is drawn, and never adjusted afterwards.

Candidate external benchmarks for a second evaluation are **[OPEN]** pending §11.

## 8. Metrics

As defined in `docs/METRICS.md`. Primary reporting object is the retention–plasticity
frontier plotted against `param_total` and against `total_algorithmic_flops`. `ppap` is a
summary of that plot and is never reported without it.

Secondary: allocation error against `K*`, spawn precision/recall, merge-loss decomposition
into decision/mechanism/interference components, recovery after merge, reuse and
reinstatement rates, zombie rate, routing entropy and specialisation NMI.

EXP-000 finding: **specialisation NMI must not be reported as evidence of good routing**,
because a fixed random partition scored higher NMI than the learned router while
performing 18 points worse.

## 9. Seeds and statistics

- Development/calibration seeds: 900–999. Confirmatory seeds are drawn from a disjoint
  range and the selection rule is recorded before the first confirmatory run.
- Minimum 5 paired independent confirmatory seeds; model-initialisation seeds paired
  across arms within each lifetime seed.
- Paired bootstrap 95% CIs on all primary comparisons.
- Every reported number carries its seed list and complete config.

## 10. Invalidation criteria

A run is invalid, archived and excluded from aggregate reporting if any validity flag in
`docs/METRICS.md` §9 fires: `ceiling`, `floor`, `taskid_leak`, `budget_breach`,
`uncounted_decision`, `unmatched_control`, `seed_reuse`. Invalid runs are never silently
dropped; the count and reasons appear in the paper.

Additional invalidation specific to this track:

- any arm whose consolidation policy reads `same_skill` or any other ground-truth field —
  those exist only for post-hoc scoring;
- any capacity-matched control whose `param_total` differs from its target by more than 2%;
- any comparison where `decision_flops` was excluded from the compute match.

## 11. **[OPEN]** — dependency on `in-c0/plasticity-routing`

The following must be resolved from that track's design lessons before this protocol is
frozen:

1. **Routing substrate.** Whether the router is density-based, learned-gated, or
   reconstruction-based materially changes what `decision_flops` means and whether O(N)
   routing is admissible at the scale tested.
2. **Plasticity measurement.** The plasticity axis of the primary frontier must use the
   same definition that track settles on, or the two papers cannot be read together.
3. **Whether routing and capacity are separable at all** in the substrate chosen. If that
   track finds they are not, H1 as written is not testable and must be reformulated.
4. **Base model, adapter family and rank**, so that `param_total` is comparable across the
   programme.

Until these are settled, this repository does development simulation, infrastructure and
literature work only.

## 12. Publication policy

- Preregister before confirmatory runs; freeze protocol version, code SHA, environment
  lock, model/tokenizer revisions and seed rule.
- Publish negative results. EXP-000's negative finding is already recorded in
  `experiments/EXP-000-TOY-RESULT.md` and will appear in any paper from this track.
- Never tune thresholds, benchmark difficulty, routing hyperparameters or the success
  criterion to make a favoured arm win. Difficulty calibration uses the
  method-independent criterion in §7 only.
- Report the count and reasons of invalidated runs.
