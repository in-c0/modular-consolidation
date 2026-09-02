# EXP-000 — CAMS-v0 development pilot

**Status: DEVELOPMENT SIMULATOR. Not evidence for or against any claim about modular
consolidation in real models.** Its purpose is to check that the protocol, the budget
accounting and the metrics behave, and to expose design faults cheaply. It is reported
here because it already falsified two things, and because negative results are preserved
in this programme whether or not they are convenient.

- Code: `scripts/run_toy.py`, `scripts/calibrate_stream.py`
- Learner: per-module ridge over frozen random features (closed form; no optimiser
  hyperparameters that could be tuned toward a favoured arm)
- Stream: CAMS-v0, `K* = 6`, 18 segments, `region_scale = 0.7`
- Seeds: 0–4 (development). Confirmatory seeds are not yet drawn.

## Finding 0 — the benchmark was invalid at its first setting

At the original `region_scale = 2.2`, the single-adapter arm reached 0.933 retention and
every modular arm landed within 1.2 points of it. The benchmark had no headroom for
modularity to matter, which is precisely the failure mode that *Dimensionality Controls
When Modularity Helps* warns about.

Difficulty was recalibrated using a **method-independent** criterion
(`scripts/calibrate_stream.py`): the gap between a single adapter and an oracle-task-ID
bank must be at least 0.15, and the single adapter must not be at ceiling. That criterion
never mentions which modular policy wins. Development seeds 900–902 were used for
calibration and are disjoint from the seeds reported below.

| region_scale | single-adapter retention | oracle-ID retention | headroom | admissible |
| --- | --- | --- | --- | --- |
| 2.20 | 0.906 | 0.953 | 0.046 | no |
| 1.20 | 0.842 | 0.916 | 0.074 | no |
| 0.90 | 0.786 | 0.892 | 0.106 | no |
| **0.70** | **0.721** | **0.875** | **0.154** | **yes (chosen)** |
| 0.50 | 0.626 | 0.854 | 0.228 | yes |

The least-difficult admissible setting was chosen, to avoid picking difficulty that
flatters modularity.

## Results, seeds 0–4

| arm | retention | plasticity | forgetting | k_final | P_total | storage | decision FLOPs | NMI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 single adapter | 0.681 | 0.714 | 0.053 | 1.0 | 192 | 11 136 | 1.0e4 | 0.00 |
| A2 fixed bank, random routing | 0.630 | 0.640 | 0.035 | 6.0 | 1 152 | 66 816 | 2.1e5 | 0.69 |
| A3 fixed bank, learned routing | 0.809 | 0.818 | 0.022 | 6.0 | 1 152 | 66 816 | 2.0e5 | 0.66 |
| A4 dynamic spawn | 0.848 | 0.850 | 0.010 | 16.0 | 3 072 | 178 176 | 3.9e5 | 0.98 |
| A5 spawn + merge | 0.795 | 0.804 | 0.018 | 8.8 | 1 690 | 97 997 | 2.1e5 | 0.70 |
| A6 spawn + merge + retire | 0.660 | 0.805 | 0.157 | 2.6 | 499 | 82 406 | 8.4e4 | 0.63 |
| C-OID(A4) oracle task ID | 0.867 | 0.865 | 0.008 | 16.0 | 3 072 | 178 176 | 0 | 1.00 |
| C-TERM(A4) | 0.848 | 0.850 | 0.010 | 16.0 | 3 072 | 178 176 | 3.9e5 | 0.98 |
| C-TERM(A6) | 0.738 | 0.753 | 0.036 | 2.6 | 499 | 28 954 | 7.9e4 | 0.24 |
| C-PEAK(A6) | 0.781 | 0.794 | 0.027 | 5.0 | 960 | 55 680 | 1.6e5 | 0.54 |
| C-RSPAWN(A4) | 0.817 | 0.820 | 0.021 | 16.6 | 3 187 | 184 858 | 3.8e5 | 0.76 |
| C-RMERGE(A5) | 0.811 | 0.818 | 0.017 | 11.5 | 2 208 | 128 064 | 2.7e5 | 0.78 |

Merge-loss decomposition (mean per merge event):

| arm | decision loss | mechanism loss | total | merge precision vs ground truth |
| --- | --- | --- | --- | --- |
| A5 spawn + merge | +0.0006 | −0.0006 | −0.0000 | 1.00 |
| A6 spawn + merge + retire | +0.0030 | −0.0005 | +0.0025 | 1.00 |
| C-RMERGE(A5) random pairs | +0.0026 | −0.0003 | +0.0023 | 0.67 |

## Finding 1 — routing has value; bad routing is worse than none

A3 (0.809) beats A2 (0.630) at identical capacity, so learned routing is doing real work.
A2 also falls *below* the single adapter (0.681): a bank with a poor router is worse than
no bank at all. Any paper reporting a modular win must therefore show its router beats
random routing at matched capacity, not merely that its bank beats a single model.

Note that A2's specialisation NMI is 0.69 — higher than A3's 0.66 — while performing far
worse. **Specialisation NMI is not evidence of a good router.** When input regions are
separable, even a fixed random partition correlates strongly with the latent skill.

## Finding 2 — dynamic allocation added nothing over choosing the right size

`C-TERM(A4)` is numerically identical to `A4`. This is not a bug: A4's cap never binds, so
capping at its own realised `k_final` reproduces the same run. The comparison is therefore
*degenerate by construction* for a spawn-only arm, which is a design lesson: the terminal
capacity control only carries information for arms that shrink, i.e. those that consolidate.
`A4` vs `C-RSPAWN(A4)` (0.848 vs 0.817) is the informative spawn comparison, and it does
support the spawn *criterion* over matched-rate random spawning.

## Finding 3 — consolidation was dominated by right-sizing (the falsification fires)

This is the result the protocol was built to detect.

- `A6` (0.660 retention, 499 parameters) vs `C-TERM(A6)` (0.738, 499 parameters): a fixed
  bank of A6's own final size beats the full spawn/merge/retire lifecycle by 7.8 points at
  **identical** capacity.
- `A5` (0.795, 1 690 parameters) vs `A3` (0.809, 1 152 parameters): a fixed bank of six
  beats spawn-then-merge at *fewer* parameters.
- `A6` vs `C-PEAK(A6)` (0.781 at 960 parameters): keeping the peak allocation would have
  been better than consolidating down to it.

On CAMS-v0 at this difficulty, with these policies, **retention is close to a monotone
function of live capacity, and consolidation buys nothing that choosing the right fixed
size would not have bought.** Transient over-allocation followed by consolidation was
*worse* than being right-sized from the start.

## Finding 4 — merge decisions were correct and it barely mattered

The criterion-driven merge selected same-skill pairs with precision 1.00 against ground
truth, versus 0.67 for merge-count-matched random pairing. Yet the accuracy cost per merge
event differed by only ~0.002. Decision quality and merge cost are, in this regime, almost
decoupled. Reporting a single "merging cost us x%" number would have hidden that entirely.

Mechanism loss is slightly *negative* (the practical operator marginally beat the exact
joint fit) because sample-weighted averaging acts as extra regularisation here. That is a
property of the closed-form learner and should not be expected to hold for gradient-trained
adapters, where the operator is the dominant source of merge damage in the literature.

## What this pilot does and does not establish

It establishes that the instrument works: the controls fire, the budget cannot be evaded,
the merge decomposition is computable, and the benchmark can be calibrated without
reference to which method wins.

It does **not** establish that modular consolidation fails in general. The toy is
deliberately friendly to merging (closed-form, near-lossless) and hostile to it in a
different way (retention tracks capacity almost linearly, so there is little interference
for consolidation to relieve). Both properties are artefacts of the learner.

What it does change is the **burden of proof**. The confirmatory experiment must be
designed against the null that capacity explains everything, and must report the
capacity-matched controls as primary rather than supplementary.

## Design faults found and fixed

1. The initial arm ladder changed two factors on its first edge (routing *and* capacity
   from A1 to A2). Caught by `tests/test_policies.py::test_adjacent_primary_arms_differ_in_exactly_one_factor`,
   not by inspection. A1 is now random routing over a bank of one.
2. Random routing was not being charged decision compute, which would have made it look
   artificially cheap.
3. Oracle-ID routing was being flagged as "uncounted decision compute" when the correct
   description is that it is handed the answer. It now carries `oracle_upper_bound`.
