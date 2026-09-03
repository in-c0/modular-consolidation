# Candidate-diversity diagnostics and prospective predictions for the real-model pilot

**Status: PREDECLARED, 2026-09-03. No real-model result exists and none may be run.** The
real-model consolidation pilot remains blocked on the common substrate export from
`in-c0/plasticity-routing` (D9). This document is written *before* any neural-model result so
that the EXP-003 mechanism becomes a set of falsifiable predictions rather than a story
fitted afterwards.

Committed before the instrumentation that records these diagnostics, following the
preregistration-before-runner convention used for EXP-003.

## 1. The sharpened question

EXP-003 replaced the generic "consolidation helps under pressure" intuition with a more
specific one. The real-model question is therefore:

> **At matched adaptation capacity, does selective consolidation create a nondominated
> retention–plasticity operating point that denial and eviction cannot provide — and is that
> advantage mediated by meaningful candidate selection rather than by merging itself?**

The second clause is the part EXP-003 makes testable. It is what separates "merging is
useful" from "merging *the right pair* is useful", and the two have different consequences
for practice.

## 2. Mechanism under test

From EXP-003 (`DEVELOPMENT_SIMULATOR`, D8):

> Useful consolidation requires **both** capacity pressure **and** enough candidate diversity
> for the merge criterion to exercise meaningful choice.

At very small active banks merging is effectively forced: with `C` live modules there are
`C(C−1)/2` admissible pairs, so at `C = 2` there is exactly one and "best" and "random"
coincide by construction. Once the candidate set is large enough, criterion-based merging
separates from random merging.

## 3. Diagnostics to record at every consolidation event

Predeclared now, to be recorded by the instrumentation and reported whether or not they
support the mechanism.

| # | Diagnostic | Definition |
| --- | --- | --- |
| C1 | `n_live` | live modules at the moment of the event |
| C2 | `n_candidate_pairs` | admissible unordered merge pairs considered, `C(C−1)/2` for an exhaustive criterion |
| C3 | `best_score` | criterion score of the selected pair |
| C4 | `second_best_score` | criterion score of the runner-up |
| C5 | `score_margin` | `best_score − second_best_score` — how decisive the choice was |
| C6 | `score_mean`, `score_std` | dispersion of the candidate score distribution |
| C7 | `same_skill` | ground-truth merge precision (synthetic streams only; scoring-time only, never read by a policy) |
| C8 | `decision_loss` | event-level, already implemented |
| C9 | `mechanism_loss` | event-level, already implemented |
| C10 | `total_merge_loss` | event-level, already implemented |
| C11 | `recovery`, `recovery_time`, `recovery_censored` | frozen-probe, already implemented |

C7 has no ground-truth analogue on a real benchmark with unknown latent skill structure. On
the standardized LM panel it is replaced by the task label as a scoring-only proxy, and that
substitution is declared here rather than chosen later.

## 4. Prospective predictions

Directional, stated before any real-model data. Each is falsifiable on its own.

**P1 — degeneracy at minimal candidate sets.** When `n_candidate_pairs = 1`, criterion-based
and random merging are indistinguishable on every event-level axis. In the simulator this is
true by construction; on a real model it is a *check on the instrumentation*, not a finding.

**P2 — monotone criterion advantage.** The advantage of criterion-based over random merging
(precision gap, and the ratio of random to criterion per-event loss) increases with
`n_candidate_pairs`.

**P3 — a minimum effective candidate-set size.** There is a candidate-set size below which
the merge-versus-random advantage is not detectable, and above which it is. P3 is confirmed
only if the transition reproduces at two different absolute skill/task counts, mirroring the
EXP-003 recurrence rule.

**P4 — margin predicts damage.** Within a given candidate-set size, larger `score_margin`
predicts lower `total_merge_loss` and higher `recovery`. This tests whether the criterion is
informative *per event*, not merely on average.

**P5 — mediation, not mere merging.** Any retention–plasticity frontier advantage of
`B-MERGE` over `B-DENY` is concentrated in the regime where P2/P3 hold. If a frontier
advantage appears where the criterion is provably inert (`n_candidate_pairs = 1`), the
advantage is *not* mediated by selection and the mechanism is wrong as stated.

**P6 — pressure direction.** Consistent with EXP-003 and *against* the original intuition,
the merge-over-deny advantage does **not** increase monotonically with capacity pressure; it
requires slack. Tightest ceilings should favour denial.

P5 and P6 are the two that would most cleanly falsify the mechanism.

## 5. Exploratory threshold — not confirmatory

`EXPLORATORY, POST-HOC.` Derived from the committed EXP-003 payload after results were
visible, and therefore **not** evidence for anything. It is recorded only so that a
prospective test exists.

Criterion advantage against admissible pair count, aggregated over the grid:

| admissible pairs `C(C−1)/2` | cells | mean precision gap | mean loss ratio (random ÷ criterion) |
| --- | --- | --- | --- |
| 1 (`C = 2`) | 2 | **−0.02** | 1.00 |
| 3–6 (`C = 3–4`) | 4 | +0.21 | 1.28 |
| 10–45 (`C = 5–10`) | 5 | +0.50 | 2.89 |
| ≥ 66 (`C ≥ 12`) | 3 | +0.64 | 3.23 |

Read off exploratorily, the transition sits somewhere above one admissible pair and the
advantage is already clear by roughly 6–10 pairs (`C ≈ 4–5`). One wrinkle is recorded rather
than smoothed: at `K*=24, C=4` the loss ratio is 0.92 — random merging was marginally
*cheaper* per event there — while the precision gap was still positive (+0.14).

**Prospective test.** The real-model pilot predeclares the threshold hypothesis as: *the
merge-versus-random advantage is not detectable at `n_candidate_pairs = 1`, and is detectable
at `n_candidate_pairs ≥ 6`.* No numeric threshold fitted to EXP-003 may be reported as
confirmatory, and the pilot may not re-tune this boundary after seeing its own results.

## 6. What would falsify the mechanism

- P5 fails: a frontier advantage appears where the criterion cannot select.
- P2 fails: criterion advantage is flat or non-monotone in candidate-set size.
- P4 fails: `score_margin` carries no per-event information about loss or recovery.
- The frontier advantage does not reproduce at all on a real model under matched capacity and
  full parameter/storage/decision-compute accounting.

Any of these is a publishable negative and is reported as the headline, per the standing
policy.

## 7. Non-claims

- This document establishes nothing. It contains no real-model result and licenses no run.
- EXP-003 remains `DEVELOPMENT_SIMULATOR`; the mechanism is a simulator mechanism, not a
  claim about neural continual learning.
- The architecture-paper gate stays **CLOSED** (D9).
- No recurrence, stream-generation, threshold or seed change is authorised by this document.
- Nothing here selects or imports a substrate. `in-c0/plasticity-routing` owns that decision
  and has not made it.
