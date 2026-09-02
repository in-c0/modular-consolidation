# EXP-001 — the interference regime, and why it could not work

**Status: DEVELOPMENT SIMULATOR — NEGATIVE INSTRUMENT RESULT.** No admissible configuration
was found. This is preserved because the *reason* it failed is the most useful thing the
track has learned so far, and because relaxing the criterion until something passed would
have been the exact failure mode the protocol exists to prevent.

- Code: `scripts/calibrate_interference.py`
- Criterion: K1–K5 in `docs/BENCHMARK-POLICY.md` (K1–K4 predeclared, K5 added by Amendment A)
- Seeds: 900–902 (development)

## Motivation

EXP-000 showed that a stream of separated input regions is solved by capacity alone. CAMS-v1
added an **interference** knob so that distinct skills share input regions and demand
conflicting mappings, with a weak context signal as the only route to telling them apart.

## Stage 1 — K1–K4 admitted everything, including the stream already known to fail

| interference | A1 single | A2 random-routed bank | C-OID oracle | K3 gap | K4 gap | admissible |
| --- | --- | --- | --- | --- | --- | --- |
| 0.00 | 0.675 | 0.587 | 0.865 | 0.190 | 0.278 | yes |
| 0.25 | 0.645 | 0.610 | 0.855 | 0.210 | 0.245 | yes |
| 0.50 | 0.676 | 0.609 | 0.875 | 0.199 | 0.267 | yes |
| 0.75 | 0.689 | 0.616 | 0.867 | 0.178 | 0.251 | yes |
| 1.00 | 0.652 | 0.528 | 0.868 | 0.217 | 0.341 | yes |

Checking K4 against EXP-000's own stream: oracle 0.867 vs random-routed bank 0.630, gap
0.237 — it would have passed there too. A criterion that admits the stream already shown to
be incapable of answering the question is not doing its job. K4 tests whether **routing**
beats capacity, which was never in doubt.

Amendment A added **K5**: over a sweep of fixed-bank caps, over-allocation must cost at
least 0.05 retention. If retention is monotone in module count, the correct policy is
trivially "allocate the maximum" and no consolidation policy can differ.

## Stage 2 — K5 failed everywhere, across three independent mechanisms

All sweeps use `A3` (fixed bank, learned routing) only — not a consolidation method.

**Mechanism 1: interference.** Capacity curves, best K5 cost 0.010.

| interference | K=1 | K=2 | K=4 | K=8 | K=16 | K=24 | K5 cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.00 | 0.675 | 0.738 | 0.801 | 0.828 | 0.839 | 0.839 | 0.004 |
| 0.50 | 0.676 | 0.725 | 0.767 | 0.827 | 0.825 | 0.825 | 0.010 |
| 1.00 | 0.652 | 0.690 | 0.718 | 0.738 | 0.750 | 0.749 | 0.002 |

**Mechanism 2: data scarcity.** Samples per segment swept from 10× to 1× the feature
dimension. An interior optimum does appear, but never reaches the threshold.

| samples/dim | K=1 | K=4 | K=8 | K=16 | K=24 | K5 cost |
| --- | --- | --- | --- | --- | --- | --- |
| 10.0 | 0.732 | 0.791 | 0.865 | 0.865 | 0.865 | 0.000 |
| 3.3 | 0.715 | 0.780 | 0.838 | 0.822 | 0.821 | 0.017 |
| 2.0 | 0.725 | 0.810 | 0.836 | 0.816 | 0.812 | **0.024** |
| 1.0 | 0.735 | 0.776 | 0.795 | 0.810 | 0.800 | 0.010 |

**Mechanism 3: soft routing.** A density-gated mixture over *all* live modules, tested
precisely because it is the mechanism by which spare modules could dilute a prediction.

| interference | K=1 | K=4 | K=8 | K=16 | K=24 | K5 cost |
| --- | --- | --- | --- | --- | --- | --- |
| 0.00 | 0.675 | 0.802 | 0.834 | 0.850 | 0.850 | **0.000** |
| 0.50 | 0.676 | 0.770 | 0.830 | 0.832 | 0.832 | 0.000 |
| 1.00 | 0.652 | 0.723 | 0.747 | 0.767 | 0.766 | 0.000 |

Soft routing changed nothing: log-density gating in 48 dimensions is so peaked that the
mixture is effectively hard. Raising the temperature until it blurs would only reproduce
`A2`, a deliberately bad router, which is not a finding.

## The finding

> **In a parameter-isolated modular system with competent routing, the retention-versus-capacity
> curve is monotone non-decreasing. A module that is never selected cannot damage a
> prediction, so spare modules cost parameters, compute and storage — but not accuracy.**

Three consequences, in increasing order of importance:

1. CAMS cannot be made to satisfy K5 by tuning, and no attempt was made to relax the
   threshold instead.
2. **Consolidation cannot improve retention at matched capacity in this architecture
   class.** There is nothing for it to fix. The track's original H1 was therefore not merely
   unlikely, it was close to analytically false — which is exactly what EXP-000 observed
   empirically without being able to explain it.
3. In the unbounded regime, consolidation's only possible benefit is moving left along the
   efficiency frontier at equal retention. That is a **compression** claim, not a
   **forgetting** claim, and the literature should state which one it is making.

## What was done about it

The question was reformulated rather than abandoned. Consolidation can only be irreducible
to capacity when capacity cannot be increased, so the track moved to the **binding-ceiling**
regime, where every arm holds an identical number of modules and differs only in how it
frees a slot. See `docs/BENCHMARK-POLICY.md` Amendment B and
`experiments/EXP-002-CEILING-RESULT.md`, which found a significant effect there.

## Provenance

Both amendments were committed **before any consolidation arm had been run on CAMS-v1**, so
neither K5 nor K6 could have been reverse-engineered toward an outcome. The git history is
the evidence:

```
00999e3  predeclare K1-K4, log owner decisions
690d69e  Amendment A: K4 insufficient, add K5
98de714  Amendment B: K5 unsatisfiable, reformulate to binding ceiling (K6)
```

The first commit containing any consolidation result on CAMS-v1 comes after all three.
