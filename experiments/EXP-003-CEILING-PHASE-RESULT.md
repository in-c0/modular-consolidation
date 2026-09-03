# EXP-003 — capacity-pressure phase diagram (result)

**`DEVELOPMENT_SIMULATOR`. This is not real-model evidence.** It is a synthetic,
closed-form ridge learner on CAMS. It does not lift EXP-100's dependency on
`in-c0/plasticity-routing` and, on its own, establishes no architecture result.

- Preregistration: `experiments/EXP-003-CEILING-PHASE-PREREG.md` (committed `74dc652`, before the runner `5b6314c`)
- Runner: `scripts/run_ceiling_phase.py` via `make ceiling-phase`, no CI
- Analysis: `scripts/analyse_ceiling_phase.py` (read-only)
- Payload: `results/ceiling_phase/rows.json` — 600 rows, **2 619 merge event records**
- Executed at git `1d1ea78b61fbcef2b03bfe96a489f153db64a855`, tracked source clean at launch
- Python 3.14.6, numpy 2.5.2, Darwin 24.6.0 arm64 (Apple M1 Max), macOS 15.6
- Source-tree sha256 `4b8d3bf013a6be896eca8f47e318d46827891bca2879fe8bdeb90d5d8317e459`
- Full provenance: `results/ceiling_phase/PROVENANCE.txt`; stdout `run.log`, stderr `run.err`

Frozen design executed exactly as preregistered: `K* ∈ {6,12,24}`,
`ceiling/K* ∈ {1/6,1/3,1/2,2/3,5/6}`, dev seeds 900–907, five arms, `segments = 3K*`,
`recur_prob = near_dup_prob = 0.30`, `region_scale = 0.7`, one stream per `(K*, seed)`
reused across every pressure cell and arm.

## 1. Validity and construction

| check | outcome |
| --- | --- |
| all 15 `(K*, ratio)` cells present | yes |
| 8 seeds × 5 arms in every cell | yes (600 rows) |
| every paired comparison on identical seed sets | yes (runner raises otherwise) |
| capacity equality across arms per cell/seed | yes — identical `param_total`, `storage_total`, `k_final` |
| NaN/inf or missing primary metric | none |
| stream diagnostics present | yes, 24 `(K*, seed)` records |
| results excluded because K6b failed | **none** — no cell fell below K6b, and K6 is diagnostic only |
| merge events preserved | 2 619, equal to `Σ n_merge` |

**Status: VALID.** Interpretation proceeds.

Two streams did not introduce all nominal `K*` skills, recorded rather than resampled:

- `K*=6, seed 905` — only **2** distinct skills realised;
- `K*=24, seed 900` — 22 of 24.

These remain in every cell and comparison. The `K*=6, seed 905` stream is a material
deviation from nominal and is one reason the `K*=6` row should be read with caution.

**Structural degeneracy at `ceiling = 1`.** Freeing a slot requires at least two live
modules, so at `K*=6, ratio 1/6` (ceiling 1) no arm can ever evict or merge and all five
collapse to identical behaviour. Every contrast there is exactly `0.0000` by construction,
not by measurement. The same mechanism makes `B-MERGE` and `B-MERGE-RAND` identical at
ceiling 2 (`K*=6, 1/3` and `K*=12, 1/6`): with two live modules there is only one possible
pair, so "best" and "random" coincide.

## 2. Primary D5 contrasts — retention and plasticity, never collapsed

Paired mean difference with paired-bootstrap 95% CI, 8 seeds. `*` = CI excludes zero.

### B-MERGE − B-DENY

| `K*` | ceiling | ratio | retention | plasticity |
| --- | --- | --- | --- | --- |
| 6 | 1 | 1/6 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| 6 | 2 | 1/3 | −0.0258 [−0.0603, +0.0092] | +0.0199 [+0.0048, +0.0338] \* |
| 6 | 3 | 1/2 | −0.0216 [−0.0357, −0.0068] \* | +0.0133 [−0.0116, +0.0403] |
| 6 | 4 | 2/3 | −0.0137 [−0.0425, +0.0130] | +0.0224 [−0.0009, +0.0504] |
| 6 | 5 | 5/6 | −0.0027 [−0.0292, +0.0242] | +0.0314 [+0.0085, +0.0539] \* |
| 12 | 2 | 1/6 | −0.0498 [−0.0749, −0.0233] \* | +0.0191 [+0.0069, +0.0334] \* |
| 12 | 4 | 1/3 | +0.0108 [−0.0197, +0.0363] | +0.0591 [+0.0298, +0.0861] \* |
| 12 | 6 | 1/2 | +0.0303 [+0.0062, +0.0576] \* | +0.0615 [+0.0495, +0.0720] \* |
| 12 | 8 | 2/3 | +0.0411 [+0.0295, +0.0525] \* | +0.0599 [+0.0435, +0.0769] \* |
| 12 | 10 | 5/6 | +0.0638 [+0.0346, +0.0965] \* | +0.0615 [+0.0374, +0.0862] \* |
| 24 | 4 | 1/6 | −0.0140 [−0.0526, +0.0221] | +0.0719 [+0.0552, +0.0871] \* |
| 24 | 8 | 1/3 | +0.0136 [−0.0076, +0.0335] | +0.0532 [+0.0285, +0.0811] \* |
| 24 | 12 | 1/2 | +0.0289 [−0.0019, +0.0609] | +0.0678 [+0.0465, +0.0912] \* |
| 24 | 16 | 2/3 | +0.0423 [+0.0096, +0.0782] \* | +0.0688 [+0.0372, +0.1017] \* |
| 24 | 20 | 5/6 | +0.0373 [−0.0034, +0.0846] | +0.0474 [+0.0189, +0.0814] \* |

### B-MERGE − B-EVICT-LRU

| `K*` | ratio | retention | plasticity |
| --- | --- | --- | --- |
| 6 | 1/6 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| 6 | 1/3 | +0.2582 [+0.1552, +0.3655] \* | −0.0628 [−0.0785, −0.0450] \* |
| 6 | 1/2 | +0.2922 [+0.1902, +0.3916] \* | −0.0477 [−0.0690, −0.0253] \* |
| 6 | 2/3 | +0.1786 [+0.1092, +0.2465] \* | −0.0318 [−0.0511, −0.0128] \* |
| 6 | 5/6 | +0.1648 [+0.1038, +0.2263] \* | −0.0094 [−0.0214, +0.0017] |
| 12 | 1/6 | +0.2596 [+0.2304, +0.2821] \* | −0.1359 [−0.1514, −0.1210] \* |
| 12 | 1/3 | +0.2599 [+0.1956, +0.3141] \* | −0.0825 [−0.1000, −0.0667] \* |
| 12 | 1/2 | +0.2798 [+0.2196, +0.3315] \* | −0.0494 [−0.0630, −0.0351] \* |
| 12 | 2/3 | +0.2603 [+0.2053, +0.3113] \* | −0.0270 [−0.0378, −0.0168] \* |
| 12 | 5/6 | +0.2385 [+0.1778, +0.2953] \* | −0.0158 [−0.0299, −0.0054] \* |
| 24 | 1/6 | +0.2540 [+0.2059, +0.2979] \* | −0.1486 [−0.1720, −0.1263] \* |
| 24 | 1/3 | +0.2660 [+0.2303, +0.2945] \* | −0.1157 [−0.1321, −0.0969] \* |
| 24 | 1/2 | +0.2546 [+0.1910, +0.3053] \* | −0.0751 [−0.0892, −0.0610] \* |
| 24 | 2/3 | +0.2197 [+0.1813, +0.2623] \* | −0.0503 [−0.0574, −0.0435] \* |
| 24 | 5/6 | +0.1885 [+0.1698, +0.2065] \* | −0.0390 [−0.0515, −0.0284] \* |

### Absolute arm means (retention / plasticity)

| `K*` | ratio | B-DENY | B-MERGE | B-EVICT-LRU | B-EVICT-RAND | B-MERGE-RAND |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | 1/6 | 0.713/0.740 | 0.713/0.740 | 0.713/0.740 | 0.713/0.740 | 0.713/0.740 |
| 6 | 1/3 | 0.754/0.770 | 0.728/0.790 | 0.470/0.853 | 0.504/0.855 | 0.728/0.790 |
| 6 | 1/2 | 0.778/0.791 | 0.756/0.804 | 0.464/0.852 | 0.525/0.853 | 0.753/0.804 |
| 6 | 2/3 | 0.785/0.797 | 0.771/0.819 | 0.593/0.851 | 0.559/0.852 | 0.771/0.818 |
| 6 | 5/6 | 0.792/0.808 | 0.790/0.840 | 0.625/0.849 | 0.602/0.853 | 0.775/0.832 |
| 12 | 1/6 | 0.659/0.693 | 0.609/0.712 | 0.349/0.848 | 0.372/0.850 | 0.609/0.712 |
| 12 | 1/3 | 0.671/0.707 | 0.681/0.766 | 0.422/0.848 | 0.416/0.852 | 0.640/0.742 |
| 12 | 1/2 | 0.708/0.742 | 0.738/0.803 | 0.458/0.853 | 0.537/0.854 | 0.684/0.782 |
| 12 | 2/3 | 0.735/0.766 | 0.776/0.826 | 0.516/0.853 | 0.598/0.851 | 0.711/0.785 |
| 12 | 5/6 | 0.749/0.776 | 0.813/0.838 | 0.574/0.854 | 0.599/0.853 | 0.726/0.807 |
| 24 | 1/6 | 0.596/0.628 | 0.582/0.700 | 0.328/0.849 | 0.335/0.851 | 0.581/0.659 |
| 24 | 1/3 | 0.650/0.680 | 0.663/0.734 | 0.397/0.849 | 0.416/0.849 | 0.611/0.729 |
| 24 | 1/2 | 0.675/0.706 | 0.704/0.773 | 0.450/0.849 | 0.461/0.849 | 0.638/0.744 |
| 24 | 2/3 | 0.706/0.729 | 0.748/0.798 | 0.529/0.848 | 0.558/0.847 | 0.669/0.783 |
| 24 | 5/6 | 0.740/0.761 | 0.777/0.809 | 0.589/0.848 | 0.590/0.847 | 0.715/0.792 |

Capacity is identical across arms within every cell by construction (e.g. `K*=24, 5/6`:
3 840 parameters and 222 720 bytes for all five arms).

## 3. Pressure versus absolute `K*`

`B-MERGE − B-DENY` retention mean difference:

| ratio | `K*`=6 | `K*`=12 | `K*`=24 |
| --- | --- | --- | --- |
| 1/6 | 0.0000 † | −0.0498 | −0.0140 |
| 1/3 | −0.0258 | +0.0108 | +0.0136 |
| 1/2 | −0.0216 | +0.0289 | +0.0289 |
| 2/3 | −0.0137 | +0.0411 | +0.0423 |
| 5/6 | −0.0027 | +0.0638 | +0.0373 |

† degenerate: ceiling 1 admits no slot-freeing operation.

**Two findings, one of which contradicts the motivating hypothesis for this sweep.**

1. **The sign of `MERGE − DENY` retention is governed by absolute `K*`, not by pressure.**
   At `K*=6` it is negative at every non-degenerate ratio. At `K*=12` and `K*=24` it is
   positive at every ratio from 1/3 upward. The boundary lies between 6 and 12 distinct
   skills, not at a particular ceiling ratio.
2. **Within a given `K*`, merge's advantage grows as the ceiling *loosens*, and at the
   tightest ratio deny wins.** At `K*=12` the difference runs −0.0498, +0.0108, +0.0303,
   +0.0411, +0.0638 across 1/6 → 5/6. D5's motivating expectation was the opposite — that
   denial would degrade fastest under the *most* pressure. On this grid the highest-pressure
   cells are exactly where denial is safest, and where merging is worst.

The mechanism is visible in §4: at ceiling 2 there is only one candidate pair, so merging is
forced and blind (precision 0.11–0.13, per-event loss ≈0.079). As the ceiling loosens the
criterion gets a real choice, precision rises to 0.81 and per-event loss falls by ~40×.
**Merging needs slack to select well; under maximum pressure it degenerates.**

## 4. Explanatory outcomes

### Forgetting by arm

| `K*` | ratio | B-DENY | B-EVICT-LRU | B-EVICT-RAND | B-MERGE | B-MERGE-RAND |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | 1/6 | 0.045 | 0.045 | 0.045 | 0.045 | 0.045 |
| 6 | 1/3 | 0.035 | 0.394 | 0.362 | 0.084 | 0.084 |
| 6 | 1/2 | 0.029 | 0.401 | 0.342 | 0.063 | 0.070 |
| 6 | 2/3 | 0.027 | 0.272 | 0.307 | 0.063 | 0.064 |
| 6 | 5/6 | 0.030 | 0.238 | 0.261 | 0.062 | 0.072 |
| 12 | 1/6 | 0.051 | 0.508 | 0.488 | 0.129 | 0.129 |
| 12 | 1/3 | 0.050 | 0.437 | 0.445 | 0.103 | 0.119 |
| 12 | 1/2 | 0.048 | 0.406 | 0.329 | 0.078 | 0.113 |
| 12 | 2/3 | 0.042 | 0.350 | 0.264 | 0.063 | 0.089 |
| 12 | 5/6 | 0.040 | 0.292 | 0.266 | 0.038 | 0.097 |
| 24 | 1/6 | 0.057 | 0.533 | 0.527 | 0.141 | 0.112 |
| 24 | 1/3 | 0.050 | 0.466 | 0.446 | 0.091 | 0.142 |
| 24 | 1/2 | 0.048 | 0.413 | 0.402 | 0.086 | 0.131 |
| 24 | 2/3 | 0.040 | 0.334 | 0.303 | 0.066 | 0.133 |
| 24 | 5/6 | 0.035 | 0.273 | 0.271 | 0.047 | 0.094 |

Eviction's forgetting reaches 0.53. `B-DENY` is the lowest-forgetting arm in 14 of 15 cells;
at `K*=12, 5/6` `B-MERGE` overtakes it (0.038 vs 0.040).

### Event-level merge diagnostics (2 619 events)

Means per event. `prec` is ground-truth merge precision, `cens` the censored fraction.

| `K*` | ratio | arm | n | decision | mechanism | total | prec | recov | rtime | cens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | 1/3 | B-MERGE | 44 | 0.0358 | 0.0431 | 0.0789 | 0.20 | 0.615 | 1.72 | 0.27 |
| 6 | 1/3 | B-MERGE-RAND | 44 | 0.0358 | 0.0431 | 0.0789 | 0.23 | 0.615 | 1.72 | 0.27 |
| 6 | 1/2 | B-MERGE | 34 | 0.0165 | 0.0088 | 0.0253 | 0.41 | 0.654 | 0.74 | 0.32 |
| 6 | 1/2 | B-MERGE-RAND | 34 | 0.0164 | 0.0187 | 0.0351 | 0.26 | 0.668 | 1.28 | 0.26 |
| 6 | 2/3 | B-MERGE | 42 | 0.0184 | 0.0100 | 0.0284 | 0.43 | 0.700 | 0.70 | 0.36 |
| 6 | 2/3 | B-MERGE-RAND | 34 | 0.0201 | 0.0101 | 0.0302 | 0.18 | 0.542 | 0.86 | 0.35 |
| 6 | 5/6 | B-MERGE | 52 | 0.0124 | 0.0054 | 0.0178 | 0.54 | 0.658 | 0.71 | 0.33 |
| 6 | 5/6 | B-MERGE-RAND | 45 | 0.0150 | 0.0126 | 0.0276 | 0.16 | 0.663 | 0.93 | 0.36 |
| 12 | 1/6 | B-MERGE | 62 | 0.0366 | 0.0420 | 0.0786 | 0.11 | 0.590 | 1.38 | 0.40 |
| 12 | 1/6 | B-MERGE-RAND | 62 | 0.0366 | 0.0420 | 0.0786 | 0.13 | 0.590 | 1.38 | 0.40 |
| 12 | 1/3 | B-MERGE | 76 | 0.0140 | 0.0076 | 0.0216 | 0.45 | 0.631 | 0.84 | 0.33 |
| 12 | 1/3 | B-MERGE-RAND | 61 | 0.0211 | 0.0165 | 0.0376 | 0.13 | 0.510 | 1.06 | 0.49 |
| 12 | 1/2 | B-MERGE | 90 | 0.0066 | 0.0027 | 0.0094 | 0.60 | 0.729 | 0.60 | 0.20 |
| 12 | 1/2 | B-MERGE-RAND | 72 | 0.0143 | 0.0093 | 0.0236 | 0.19 | 0.531 | 1.14 | 0.42 |
| 12 | 2/3 | B-MERGE | 88 | 0.0042 | 0.0018 | 0.0059 | 0.69 | 0.805 | 0.36 | 0.16 |
| 12 | 2/3 | B-MERGE-RAND | 73 | 0.0100 | 0.0051 | 0.0151 | 0.15 | 0.489 | 1.12 | 0.45 |
| 12 | 5/6 | B-MERGE | 89 | 0.0012 | 0.0008 | 0.0021 | 0.81 | 0.869 | 0.38 | 0.12 |
| 12 | 5/6 | B-MERGE-RAND | 90 | 0.0092 | 0.0029 | 0.0122 | 0.17 | 0.451 | 1.24 | 0.43 |
| 24 | 1/6 | B-MERGE | 149 | 0.0146 | 0.0062 | 0.0208 | 0.28 | 0.494 | 1.16 | 0.39 |
| 24 | 1/6 | B-MERGE-RAND | 113 | 0.0128 | 0.0064 | 0.0191 | 0.14 | 0.504 | 1.07 | 0.40 |
| 24 | 1/3 | B-MERGE | 126 | 0.0054 | 0.0018 | 0.0072 | 0.63 | 0.667 | 0.61 | 0.29 |
| 24 | 1/3 | B-MERGE-RAND | 167 | 0.0100 | 0.0042 | 0.0142 | 0.10 | 0.477 | 1.03 | 0.45 |
| 24 | 1/2 | B-MERGE | 175 | 0.0049 | 0.0010 | 0.0059 | 0.60 | 0.699 | 0.65 | 0.27 |
| 24 | 1/2 | B-MERGE-RAND | 168 | 0.0086 | 0.0032 | 0.0118 | 0.08 | 0.478 | 1.18 | 0.45 |
| 24 | 2/3 | B-MERGE | 164 | 0.0022 | 0.0007 | 0.0029 | 0.76 | 0.727 | 0.61 | 0.23 |
| 24 | 2/3 | B-MERGE-RAND | 161 | 0.0074 | 0.0024 | 0.0099 | 0.09 | 0.370 | 1.12 | 0.53 |
| 24 | 5/6 | B-MERGE | 151 | 0.0012 | 0.0003 | 0.0015 | 0.79 | 0.795 | 0.55 | 0.16 |
| 24 | 5/6 | B-MERGE-RAND | 153 | 0.0053 | 0.0011 | 0.0064 | 0.07 | 0.470 | 0.80 | 0.48 |

**The merge criterion is no longer inert.** EXP-002 found precision 0.50 vs 0.36 and
statistically indistinguishable aggregates. Here, at `K*≥12` with ratio ≥1/3, the criterion
separates cleanly from random pairing on every event-level axis at once: precision
0.45–0.81 vs 0.07–0.19; per-event total loss 3–8× lower; recovery 0.63–0.87 vs 0.37–0.53;
recovery time roughly halved; censoring 0.12–0.33 vs 0.42–0.53. Absolute per-event loss
falls to 0.0015 at `K*=24, 5/6`.

Both loss components shrink together — the criterion reduces `decision_loss` (merging the
right pair) *and* `mechanism_loss` (similar modules are cheaper to pool). The EXP-000/EXP-002
observation that decision quality and outcome are weakly coupled therefore **does not
generalise**: it was a property of the small-`K*`, tight-ceiling regime where merging is
forced and blind.

## 5. Does a reproducible Pareto-extending region exist?

The predeclared rule is applied mechanically. Two readings of it diverge, and the divergence
is reported rather than resolved unilaterally.

**Reading A — strict, as literally implemented.** Expansion requires a non-negative *mean*
on both primary axes plus a significant gain on at least one, **against deny and against
evict separately**.

> Result: **no cell qualifies.** `B-MERGE` never dominates `B-EVICT-LRU`, because eviction's
> plasticity is significantly higher in 14 of 15 cells.
> Mechanical verdict: **OPERATING_POINT**.

**This clause is unsatisfiable by construction.** `B-EVICT-LRU` installs a fresh module on
every admission, so it maximises plasticity by definition; no arm can dominate it on that
axis. A criterion that cannot be met by any possible policy carries no information, and its
failure here is not evidence about merging.

**Reading B — standard Pareto extension against the baseline set `{deny, evict}`.** Merge
extends the frontier if it is dominated by neither baseline and strictly dominates at least
one.

> Result: **a reproducible region exists.** `B-MERGE` strictly dominates `B-DENY` on both
> retention and plasticity — with at least one axis significant, and never a significant
> loss on the other — in 8 cells:
>
> - `K*=12` at ratios 1/3, 1/2, 2/3, 5/6
> - `K*=24` at ratios 1/3, 1/2, 2/3, 5/6
>
> These are contiguous in pressure (three adjacent-ratio runs at each `K*`) and every one of
> the four ratios is replicated at two absolute `K*` values. Merge is dominated by neither
> baseline anywhere. The reproducibility requirements in the preregistration — adjacency
> **and** replication at a second absolute `K*` — are both satisfied.

The clearest single cell is `K*=12, ratio 5/6`: `B-MERGE` 0.813 retention / 0.838
plasticity against `B-DENY` 0.749 / 0.776 — better on both axes by a wide margin, at
identical capacity, with `B-EVICT-LRU` at 0.574 / 0.854.

**This divergence is an owner decision, not mine.** Reading A is what the preregistration
literally says; Reading B is the standard meaning of "extends the deny/evict Pareto
frontier" in D5 and is the only one of the two that is satisfiable. Both are reported in
`results/ceiling_phase/analysis.json` under `pareto_extending` and
`baseline_frontier_extension`. No cell was selected after the fact; both readings are
computed over the complete grid.

## 6. Architecture-paper consequence, stated mechanically

D4 keeps the architecture paper conditional and re-opens it only on a reproducible region
that expands the retention–plasticity frontier relative to deny **and** evict at identical
capacity, with the real-model result still required afterwards.

- Under **Reading A**, the condition is not met and the architecture paper **stays closed**.
  EXP-003 remains a methods-paper operating-point result.
- Under **Reading B**, the simulator-level condition **is** met: a contiguous, twice-replicated
  region in which merging dominates the strongest missing baseline on both primary axes at
  identical capacity.

Under either reading, **nothing here re-opens EXP-100 by itself.** D4 requires a real-model
result surviving capacity/storage/decision-compute accounting, and that remains gated on
`in-c0/plasticity-routing`. EXP-100 was not run and its status is unchanged.

Stated without inflation: this is a synthetic result, on one closed-form learner, on one
stream family, at development seeds only. It is a reason to *ask* the D4 question, not an
answer to it.

## 7. Threats to validity

- Closed-form ridge learner: merging is near-lossless here, and `mechanism_loss` is
  correspondingly small. Gradient-trained adapters are expected to make the merge operator a
  much larger share of the damage, and HARC (arXiv:2606.03391) shows merging also breaks
  routing, which this substrate cannot exhibit.
- `K*=6, seed 905` realised only 2 distinct skills; `K*=24, seed 900` realised 22 of 24.
  Both are retained. The `K*=6` row is the weakest in the grid for this reason.
- `ceiling = 1` is structurally degenerate and `ceiling = 2` collapses `B-MERGE` onto
  `B-MERGE-RAND`; three of the fifteen cells therefore carry no criterion information.
- Development seeds only, 8 per cell. No confirmatory seed set has been drawn.
- LRU is one eviction rule; a stronger rule would narrow the merge–evict retention gap.
- The absolute-`K*` boundary is located only between 6 and 12; the grid does not resolve it
  further, and no additional `K*` was added after seeing the curves.

## 8. What was not done

- No generator, recurrence, threshold, seed or arm was altered in response to the curves.
- No cell was excluded, and no result was dropped for failing K6b.
- No subset was inspected before the complete grid existed.
- EXP-100 was not run.
