# M6 Gate R — verdict

## `M6 NATIVE REPRODUCTION: FAILED`

Executed 2026-09-10 under owner authorization. Stage 1 only: `T-FULL` × 5 seeds, official
config `bsmnist_2l_noracl.yaml` unmodified, official checkout pinned at
`aa0014c8478b18e70420d3ac451d4e4472ff7040` and verified clean. All five seeds completed
(exit 0) in 7m31s total, ~90s per run — considerably faster than the ~6 min estimated in the
compute dossier.

Criteria were frozen in the preregistration (`d523dde`) before any result existed. They are
applied here exactly as written, with no seed excluded.

| seed | observed acc | published acc | delta | params | w0 | w1 | grew |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 82.80 | 75.17 | +7.63 | 9 576 | 12 | 12 | **no** |
| 1 | 67.82 | 73.53 | −5.71 | 9 646 | 12 | 17 | yes |
| 2 | 82.23 | 71.26 | +10.97 | 9 576 | 12 | 12 | **no** |
| 3 | 83.19 | 71.95 | +11.24 | 9 604 | 12 | 14 | yes |
| 4 | 74.79 | 74.27 | +0.52 | 9 604 | 12 | 14 | yes |

Mean accuracy **78.17** vs published 73.24 (**+4.93**).
Mean parameters **9 601** vs published 19 255 (**−50.1%**).

| criterion | threshold | result |
| --- | --- | --- |
| R1 mean accuracy within ±2.0 | ±2.0 of 73.24 | **FAIL** (+4.93) |
| R2 mean parameters within ±15% | ±15% of 19 255 | **FAIL** (−50.1%) |
| R3 every seed within ±5.0 | ±5.0 | **FAIL** (3 of 5 seeds outside) |
| R4 growth occurred in every seed | ≥1 growth event | **FAIL** (2 of 5 seeds never grew) |

**All four criteria fail.** Per the preregistration, published-gain attribution does **not**
proceed as though the implementation were validated. No attribution arm has been run and none
will be until this is resolved.

## The failure has a clear shape

It is **not** noise and it is **not** a small offset. Growth is systematically under-firing:
the network barely expands, ending at roughly **half** the published parameter count, and in
two seeds it never grows at all.

Instrumented diagnosis of seed 0 (reporting-only, official code unmodified):

- `neurogenesis_step` was entered **210 times**; the ED pre-check reported "did not fire"
  **0 times**, so the Effective Dimension half of the trigger was satisfied on essentially
  every evaluation;
- yet the step returned no growth on all 210 occasions;
- so the decline happens **inside** `neurogenesis_step` — at the Fisher-saturation gate or in
  the ED-derived sizing — not at the ED pre-check.

Observed per-layer Fisher thresholds are ~4e-5 (layer 0) and ~1.9e-4 (layer 1) at task 0,
rising roughly 2× per task; all state is float32.

**Accuracy is higher than published in four of five seeds while capacity is half.** A smaller
network outperforming the published larger one is itself an anomaly that argues against
dismissing this as ordinary run-to-run variation.

## Leading hypothesis — environment, not configuration

The source declares `jax>=0.4.28`, a 2024-era floor. This run used **JAX 0.11.1**. Between
those versions the default PRNG implementation, SVD backend and reduction behaviour can all
differ, and every one of those feeds the ED statistic (`trigger_act` counts singular values
above `orth_thresh`) and the Fisher diagonal. A shifted ED or Fisher scale would change the
growth decision without any config being touched.

This is a **mechanical environment hypothesis** and testing it is permitted. What is *not*
permitted, and has not been done: altering learning rate, thresholds, epochs, initialization
scale, the EWC coefficient, seeds, the tolerance, or the benchmark. No configuration value has
been changed, and none will be to close this gap.

## Status of these artefacts

Immutable. This verdict and the five run directories are committed as observed. Any subsequent
environment-diagnosis run goes to a **separate** results directory and does not overwrite these.
