# M6 — NORACL native re-analysis, preregistration

**Status: PREREGISTERED / UNRUN. No result-bearing M6 run has been executed and none is
authorised by this document.**

> **Superseded in part by Amendment M6-A at the end of this file (2026-09-03).** The `I-*`
> arms as described in §B2 did not isolate initialization; read §B2 together with the
> amendment, which splits them into `I-POLICY-*` and `I-REPLAY-*`. Committed before the runner and instrumentation, following the
repository's preregistration-before-runner rule.

Layer B (`experiments/NATIVE-FIDELITY-LEDGER.md`): this asks whether NORACL's *published
gain* survives faithful reconstruction plus the missing attribution controls, on its **own
native substrate**. It is not a standardized transplant and must never be reported as one.

## 0. Pinned sources

| Item | Value |
| --- | --- |
| Paper | Raghunathan, Metzner, Kriener, Payvand — *NORACL: Neurogenesis for Oracle-free Resource-Adaptive Continual Learning*, arXiv:**2604.27031v1** (Preprint) |
| Official repository | `github.com/karthik-charan/NORACL` |
| **Pinned revision** | `aa0014c8478b18e70420d3ac451d4e4472ff7040` (branch `main`, 2026-08-12T09:11:45Z) |
| Licence | MIT |
| Framework | **JAX** + optax (`jax>=0.4.28`, `jaxlib>=0.4.28`, `optax>=0.2.2`); torch/torchvision used only for dataset loading |
| Entrypoint | `train.py --config <name-or-path> --seed <n> --results_dir <dir>` |

All statements below were read from that revision or from the paper PDF, not from the ledger
summary.

### Two identifier corrections to the ledger, from direct source inspection

The ledger's Layer-B entry listed control names taken from **config filenames**. The
authoritative identifiers in the code are different, and this preregistration uses the code's:

| Ledger said | Source truth | Evidence |
| --- | --- | --- |
| trigger `fsat_only` | trigger is **`fisher_only`**; `fsat_only` is only a *filename* fragment (`configs/bsmnist_2l_noracl_fsat_only.yaml` sets `growth_trigger: fisher_only`) | `noracl/core/growth.py::GROWTH_TRIGGERS`, config body |
| trigger `random` | **`random` is an initialization mode, not a trigger.** `configs/bsmnist_2l_noracl_random.yaml` sets `growth_trigger: ed_fisher, init: random` | config body, `noracl/core/init.py::INIT_STRATEGIES` |

Authoritative sets:

```python
GROWTH_TRIGGERS = ("ed_fisher", "ed_only", "fisher_only", "loss_plateau", "fixed_pertask")
INIT_STRATEGIES = ("qr_init", "he_normal", "xavier", "random", "nullspace", "zero", "vp_zfo")
```

**There is no random-timing growth trigger in the official code.** Consequence for Factor T
is handled in §B1.

---

## A1. Frozen native reproduction target

**Benchmark: Binary Split MNIST, 2 hidden layers.** Config `configs/bsmnist_2l_noracl.yaml`.
Chosen because it is the cheapest native benchmark (5 tasks), it has a matched static
baseline shipped, the complete trigger/init ablation family exists for exactly this
`bsmnist_2l` stem, and its published per-seed values are shipped in the repository.
Permuted MNIST 2L is named as the **secondary replication** benchmark and is not part of the
primary run.

| Field | Value | Source |
| --- | --- | --- |
| Dataset | `binary_split_mnist`, `n_in: 784`, `n_out: 2` | config |
| Tasks | `n_tasks: 5` (5 binary digit-pair tasks) | config |
| Architecture | MLP, `n_layers: 2`, `hidden_dim: 12` initial | config |
| Epochs | `n1_epochs: 10` on task 1, `n_epochs: 30` on every later task | config, `generate_configs.py` Table-3 block |
| Optimizer | optax; `lr_1: 0.1` on task 1, `lr: 0.005` later; `lr_boost_multiplier: 3.0`; **optimizer state re-initialised after every growth event** (`opt_state = optimizer.init(params)`) | config, `noracl/training/loop.py` |
| Batch size | 256 | config |
| EWC | `ewc: true`, `importance: 5000`, Fisher EMA `alpha: 0.9`, `inherit_output_fisher: false` | config |
| ED settings | `orth_thresh: 0.05` (ε, SVD singular-value threshold), `gamma: 0.9` (γ, ED discount); `φ_l = trigger_act(acts) / M_l` | config, `noracl/core/ed.py` |
| Fisher gate | `f_sat_percentile: 25` (p) against a running per-layer threshold `τ_l` | config, `noracl/core/growth.py` |
| Growth trigger | `growth_trigger: ed_fisher` — ED fires when `any(φ_curr[l] > γ·φ_0[l])` over hidden layers `l < len(params)-1`; Fisher gate then applies inside `neurogenesis_step` | `loop.py`, `growth.py` |
| Cool-down | `annealing_epochs: 3` (C) after each growth event | config |
| Initialization | `init: qr_init`, `qr_init_scale: 0.2` (s_init); fan-out zero; Fisher/EWC anchors zero-padded | config, `noracl/core/init.py`, `noracl/core/fisher.py::pad_to_shape` |
| Fixed step | `k_fixed: 2` (used only by triggers that do not size from ED) | config |
| Other | `variance_preserve: false` | config |
| Seeds | config default `seed: 0`; the shipped published runs use **s0–s4**, so the reproduction uses seeds **0,1,2,3,4** | `results/paper/*_s{0..4}` |
| Metric | `avg_accuracy` per task boundary, plus `width_l0`, `width_l1`, `total_params` | `results/paper/.../per_task.csv` |
| Preprocessing | as implemented in `noracl/data/mnist.py` at the pinned revision; not modified | code |

### Published targets — traceable config↔result pairs

The repository ships the published per-seed measurements. These are the reproduction target
because they are traceable to a named config, unlike a table cell.

`results/paper/bsmnist_2l_noracl_s{0..4}/per_task.csv`, final task row:

| seed | final `avg_accuracy` | `total_params` |
| --- | --- | --- |
| 0 | 75.17 | 19 804 |
| 1 | 73.53 | — |
| 2 | 71.26 | — |
| 3 | 71.95 | — |
| 4 | 74.27 | — |
| **mean ± sd** | **73.24 ± 1.62** | **19 255 ± 456** |

Matched static baseline `bsmnist_2l_static32_s{0..4}`: **69.75 ± 3.39** at **26 176** params
(fixed).

### Discrepancy recorded before choosing anything

The paper's Table 1 reports Binary Split MNIST 2L NORACL as **73.9 ± 2.5 at 23.3k ± 3.1
params**. The shipped per-seed values give **73.24 ± 1.62 at 19.26k ± 0.46k**. Accuracy is
close; **the parameter counts differ materially (23.3k vs 19.3k)**. The same pattern appears
on Permuted MNIST 2L: Table 1 reports 79.4 ± 0.7, the shipped values give 77.41 ± 2.08 at a
comparable parameter count.

This is **not resolved here**, and no attempt is made to decide which is "the" published
number. The reproduction gate targets the **shipped per-seed values**, because those are an
explicit config↔result pair. The Table 1 comparison is reported as a secondary observation.
If the reproduction matches the shipped values but not Table 1, that is reported as a
finding about the source's internal consistency, not as a reproduction failure.

Also recorded: `results/paper/*_noracl_s*/config.json` are **stubs** carrying
`"source": "published values (see PROVENANCE.md)"` with only 8 fields, whereas the
`*_static*` directories contain full run outputs (`per_epoch.csv`, `acc_matrix.csv`,
`summary.json`, `timing.csv`). So the NORACL rows are shipped published values; the static
rows are shipped run artefacts. The asymmetry is noted, not corrected.

---

## A2. Two gates, in order

### Gate R — native reproduction

> Can the official NORACL configuration, at the pinned revision, reproduce its own published
> native result closely enough that attribution is meaningful?

**Reproduction tolerance — `OURS`, introduced by us because the source states none.**
Frozen here, before any run:

- **R1 (accuracy):** the 5-seed mean final `avg_accuracy` lies within **±2.0 points absolute**
  of 73.24.
- **R2 (capacity):** the 5-seed mean `total_params` lies within **±15%** of 19 255.
- **R3 (per-seed sanity):** no individual seed differs from its published counterpart by more
  than **±5.0 points absolute**.
- **R4 (growth occurred):** every seed records at least one growth event, and final widths
  exceed the initial `hidden_dim: 12` on at least one layer.

±2.0 is a little over one published standard deviation (1.62), chosen to tolerate
JAX/hardware nondeterminism without tolerating a different algorithm. ±15% on capacity is
deliberately looser than the accuracy band because growth counts are integer-valued and
discretely sensitive.

**If Gate R fails**, the attribution experiment is labelled `NATIVE REPRODUCTION FAILED`, and
published-gain attribution does not proceed as though the implementation were validated.
Mechanical debugging — environment, dataset plumbing, dtype, seeding — remains allowed.
**Outcome-guided hyperparameter rescue does not.** No config value in A1 may be altered to
close a reproduction gap.

### Gate A — attribution (conditional on Gate R passing)

> Does NORACL's gain arise from **when** it grows, from **how** newly inserted capacity is
> initialized, or primarily from **ending with more capacity**?

---

## B. Causal control lattice

All arms use the pinned revision, the same dataset, order, epochs, optimizer, EWC settings
and evaluation. Only the named factor changes.

### B1. Factor T — trigger quality

| Arm | Config | Changes from NORACL |
| --- | --- | --- |
| `T-FULL` | `bsmnist_2l_noracl` | — (this is the primary arm) |
| `T-ED` | `bsmnist_2l_noracl_ed_only` | `growth_trigger: ed_only` — removes the Fisher-saturation gate |
| `T-FISHER` | `bsmnist_2l_noracl_fsat_only` | `growth_trigger: fisher_only` — removes ED sizing, fixed step `k_fixed` |
| `T-COUNT` | **derived, see below** | `growth_trigger: fixed_pertask` with `k_fixed` derived from NORACL's realised growth |

**The growth-count-matched control.** The official code has no random-timing trigger, so the
count-matched control is built from `fixed_pertask`, which fires exactly once per task with a
fixed step `k_fixed`. Following the repository convention established in EXP-003 (`C-TERM`),
its configuration is a **deterministic function of the target arm's own realised manifest**
and is computed **without reference to any test performance**:

> For each seed, let `G` be the total number of neurons NORACL added over the run (final
> total width minus initial total width, summed over hidden layers) and `T` the number of
> tasks after the first. Set `k_fixed = round(G / (T · L))` where `L` is the number of
> growable hidden layers, clipped to `≥ 1`.

This matches the **amount** grown while removing all **timing and layer-selection**
information, so `T-FULL` vs `T-COUNT` separates *grew the right amount* from *grew at the
right moments*. `k_fixed` is derived per seed from that seed's NORACL run and frozen before
`T-COUNT` executes. **`k_fixed` may not be adjusted to make `T-COUNT` resemble NORACL.**

`loss_plateau` is a **secondary/exploratory** arm, not primary: it changes both timing and
the information source, so it does not isolate a single factor.

### B2. Factor I — insertion initialization

At `growth_trigger: ed_fisher` (identical trigger events, identical realised capacity path):

| Arm | Config | `init` |
| --- | --- | --- |
| `I-QR` | = `T-FULL` | `qr_init` (NORACL's function-preserving insertion) |
| `I-RANDOM` | `bsmnist_2l_noracl_random` | `random` |
| `I-XAVIER` | `bsmnist_2l_noracl_xavier` | `xavier` |

`he_normal`, `nullspace`, `zero` and `vp_zfo` are **secondary/exploratory**. `I-RANDOM` is the
minimal contrast (ordinary initialization) and `I-XAVIER` a standard alternative; two suffice
to identify the factor.

Note: fan-out is zero for every insertion path, so this factor isolates the **fan-in**
initialization, not function preservation in its entirety. Stated here so the result is not
over-read.

### B3. Factor C — capacity

| Arm | Config | Role |
| --- | --- | --- |
| `C-PAPER` | `bsmnist_2l_static32` | the paper's own static baseline, `hidden_dim: 32`, 26 176 params |
| `C-MATCH` | **derived static** | `grow: false`, `hidden_dim` chosen per seed to match NORACL's realised final `total_params` |

`C-PAPER` is **not** capacity-matched — at 26 176 params it is *larger* than NORACL's realised
~19.3k, which favours the static baseline and is worth reporting as such. `C-MATCH` is the
control the paper does not run.

**Derivation of `C-MATCH`, frozen before execution.** Static configs express one
`hidden_dim` for both layers, while NORACL grows layers independently (e.g. seed 0 ends at
24/38). An exact per-layer width match is therefore not expressible natively. So:

> For each seed, choose the integer `hidden_dim` minimising `|total_params(static, hidden_dim)
> − total_params(NORACL final, that seed)|`, ties broken to the smaller value. Report the
> residual parameter mismatch for every seed.

This is native — same code path, same config schema, only a width value — and it is derived
from NORACL's own realised capacity, never from test performance.

**Capacity trajectory, not just final width.** A static model has its full capacity from step
one, while NORACL acquires capacity over time, so final-width equality is not complete
accounting. Every arm therefore reports:

- initial parameter count;
- final parameter count;
- **parameter-time integral** — `Σ_epochs params(epoch)`, i.e. capacity exposure over training;
- bytes (parameters + EWC/Fisher state);
- training FLOPs and inference FLOPs under a declared cost model.

`C-MATCH` will have a strictly larger parameter-time integral than NORACL by construction.
That is the point: it makes explicit that dynamic growth is *cheaper in capacity-exposure*
even when final capacity is equal.

### B4. Primary arm set

Seven arms, each with a distinct causal role. Nothing else is promoted into the primary set.

| Arm | Identifies |
| --- | --- |
| `T-FULL` | full NORACL (also the Gate-R arm) |
| `T-ED` | contribution of the Fisher gate |
| `T-FISHER` | contribution of ED sizing |
| `T-COUNT` | timing vs amount |
| `I-RANDOM` | contribution of QR fan-in initialization |
| `I-XAVIER` | initialization robustness |
| `C-MATCH` | capacity vs adaptive allocation |

`C-PAPER` is included as a **published-baseline reference**, not as a causal arm.
Secondary/exploratory, run only if authorised separately: `loss_plateau`, `he_normal`,
`nullspace`, `zero`, `vp_zfo`, `static16`, `static64`, 1-layer variants, Permuted/Rotated
MNIST, Binary Split CIFAR-10.

### B5. Hypotheses, directional, fixed before results

- **H6.1 — Trigger irreducibility.** At matched final capacity and initialization, `T-FULL`
  achieves higher final average accuracy than `T-COUNT`, paired across seeds, if *when* to
  expand contributes causally.
- **H6.2 — Initialization irreducibility.** At matched trigger events and capacity, `I-QR`
  shows lower post-growth transient degradation and/or lower forgetting than `I-RANDOM`, if
  fan-in initialization contributes independently of growth itself.
- **H6.3 — Capacity null.** If `C-MATCH` equals `T-FULL` within the paired CI on final
  accuracy, NORACL's gain is attributed primarily to **ending wider**, not to adaptive
  growth. Capacity-exposure accounting is then the remaining defensible claim.
- **H6.4 — Timing null.** If `T-COUNT` with matched growth amount equals `T-FULL`, the
  adaptive trigger is **not required** for the measured gain.

**None of these is required to favour NORACL.** H6.3 and H6.4 are clean nulls and are valid
methods-paper results. NORACL is not a target to debunk; it is the first method whose source
is complete enough to attribute honestly, which is why it is first.

### B6. Statistics

Paired across the five seeds; paired-bootstrap 95% CIs (`metrics.paired_bootstrap_ci`,
20 000 resamples, as used in EXP-002/EXP-003). Retention and plasticity-style axes are
reported separately and never collapsed into one score — the EXP-003 lesson, where a
retention null concealed two significant opposing effects.

---

## C. Event-level accounting

EXP-003 showed aggregate outcomes conceal mechanism, so growth events are preserved
individually. For every growth event, where available:

`seed · task · epoch · layer · width_before · width_after · params_before · params_after ·
phi_curr · phi_0 · gamma · ed_threshold (γ·φ_0) · ed_fired · fisher_percentile_stat ·
fisher_tau · fisher_fired · plateau_stat · trigger_mode · init_mode · k_requested ·
k_applied · pre_growth_eval · post_growth_eval · recovery_trajectory · recovery_fraction ·
recovery_time · recovery_censored · function_preservation_error · param_delta ·
bytes_delta · flops_delta`

**Instrumentation is reporting-only.** It wraps the official functions, records their inputs
and outputs, and returns the original result unchanged. Decision logic, growth sizing,
initialization and metrics are not touched. Derived flags (`ed_fired`, `fisher_fired`) are
computed **in analysis** from recorded inputs, not reimplemented inside the hot path, so the
instrumentation cannot diverge from the method.

`function_preservation_error` is measured as the change in network output on a fixed probe
batch immediately before and after insertion. For a correct zero-fan-out insertion it must be
~0; it is recorded as a **validity check on the implementation**, not as a scientific result.

Regression tests must demonstrate that instrumentation does not change growth decisions,
inserted widths, initializations or metrics, by comparing instrumented against uninstrumented
output under a fixed seed on a deterministic miniature run.

---

## D. Score-free mechanical smoke

Permitted before authorization. Must not be able to answer any hypothesis in B5:

1–2 tasks, very few epochs, tiny config; verifies dataset and model loading, that growth can
occur, that each selected trigger path executes, that each selected init path executes, event
serialization, deterministic seeding, and that accounting fields are populated. Any smoke
output is labelled **`MECHANICAL VALIDATION — NOT SCIENTIFIC EVIDENCE`** and committed
separately. Comparative end-performance across scientific arms is **not** computed or
summarised, and the complete benchmark is not run on even one seed.

---

## E. Non-claims and boundaries

- Nothing here is evidence about NORACL. No result-bearing run has occurred.
- This does not license a Layer-A transplant. Replacing NORACL's in-layer neuron insertion
  with a LoRA expert bank remains **invalid** and may not be called NORACL.
- This does not affect the simulator track. EXP-000–EXP-003 outputs are untouched.
- The architecture-paper gate stays **CLOSED** (D9); this is Layer-B methods work.
- Launching the scientific run is an **owner decision**.

---

# Amendment M6-A — initialization was not causally isolated (2026-09-03, pre-execution)

**Committed before any M6 scientific result exists.** Raised in review via the CCS ChatGPT
thread and adopted here **on its merits**, not on the reviewer's authority — the objection is
correct and the original text was wrong.

## The defect

§B2 above says the `I-*` arms hold "identical trigger events, identical realised capacity
path". **That is not guaranteed and the claim is withdrawn.**

Changing the insertion initialization changes the values of the newly inserted fan-in weights.
Those weights change post-growth activations, which change the Effective Dimension statistic
`φ_l = trigger_act(acts)/M_l`, which is exactly what the ED half of the trigger tests. They
also change gradients and therefore the Fisher diagonal, which is what the Fisher-saturation
half tests. So altering `init` can change:

- whether the trigger fires in a given epoch;
- which layer is selected for growth;
- how many neurons are inserted;
- the number and timing of later growth events;
- the final width and final parameter count.

A source-native config with `init: random` therefore measures the **total policy effect** of
changing initialization, *including any downstream change to the growth trajectory*. It does
not isolate the direct contribution of the inserted representation itself.

## The correction — two distinct initialization analyses

### `I-POLICY-*` — native policy ablation (unchanged official configs)

`I-POLICY-QR` (= `T-FULL`), `I-POLICY-RANDOM` (`bsmnist_2l_noracl_random`),
`I-POLICY-XAVIER` (`bsmnist_2l_noracl_xavier`), all at `growth_trigger: ed_fisher`.

Interpreted **only** as: *what is the total downstream effect of replacing NORACL's
initialization policy while the adaptive growth mechanism responds normally?* No claim of
matched trigger events or matched capacity. Their realised growth-event count, timing, layers,
neurons added, final widths, final parameters and parameter-time integral are reported, and
any divergence **is part of the policy effect**, not noise to be excused.

### `I-REPLAY-*` — direct causal intervention

**`NATIVE-COMPATIBLE CAUSAL INTERVENTION — NOT AN OFFICIAL NORACL CONFIG.`** This is a Layer-B
attribution control we construct; the repository does not ship it, and it must never be
described as a NORACL configuration.

Per seed: run `T-FULL`, save a **growth manifest** containing training-time mechanism
information only — task index, epoch/step, layer, neurons inserted, and any source-behaviour
optimizer-reset point. **No evaluation outcome of any kind may enter the manifest.** Then
replay that exact manifest in separate runs changing **only** the fan-in initialization:
`I-REPLAY-QR`, `I-REPLAY-RANDOM`, `I-REPLAY-XAVIER`.

The replay runner holds fixed: dataset and order, seed, initial parameters, optimizer
settings, EWC/Fisher updates, training schedule, growth times, growth layers, neurons added,
optimizer-reset behaviour, and the full capacity trajectory. In replay mode the alternative
run's own ED/Fisher statistics **do not** decide whether growth happens — holding NORACL's
realised growth path fixed is precisely the intervention.

### Replay validity gate — must pass before any scientific use

On a miniature deterministic run:

1. `I-REPLAY-QR` reproduces `T-FULL`'s trajectory and numerical outputs to the strongest
   practical deterministic tolerance;
2. replay records exactly the same growth-event sequence;
3. replay has exactly the same width and parameter trajectory;
4. switching QR → random/xavier changes **no** scheduled event metadata;
5. no evaluation score enters the manifest or the event scheduler.

**If `I-REPLAY-QR` does not reproduce `T-FULL`, the replay intervention is invalid and the
scientific run stops before using it.**

## Hypotheses — H6.2 splits

- **H6.2a — initialization *policy* effect.** Replacing QR with native `random`/`xavier` while
  the normal adaptive trigger runs changes continual-learning performance. Source-native; may
  include downstream trajectory change.
- **H6.2b — *direct* initialization effect.** With the `T-FULL` growth manifest held fixed, QR
  outperforms random/xavier if the insertion representation contributes beyond growth timing,
  layer selection, amount and capacity. This is the stronger causal test.

A live possible outcome: **policy arms differ, replay arms do not** — meaning initialization
matters mainly through its effect on *subsequent growth decisions*, not through the inserted
representation. The reverse pattern is equally meaningful. Neither is a failure.

## Attribution rules tightened

- "Direct initialization is irreducible" may be concluded **only** from `I-REPLAY`
  comparisons. It may **not** be inferred from the policy arms alone.
- "Initialization acts through growth policy" is supported if `I-POLICY` trajectories diverge
  materially while `I-REPLAY` effects are small.
- If final accuracy is comparable but NORACL reaches it at a substantially lower
  parameter-time integral, that is reported as **resource-efficiency evidence, not an accuracy
  gain**.

## Primary arm set, revised

`T-FULL`, `T-ED`, `T-FISHER`, `T-COUNT`, `I-POLICY-RANDOM`, `I-POLICY-XAVIER`,
`I-REPLAY-QR`, `I-REPLAY-RANDOM`, `I-REPLAY-XAVIER`, `C-MATCH`; plus `C-PAPER` as the
source's static reference. `T-FULL`, `I-POLICY-QR` and `I-REPLAY-QR` are the same reference
where equivalent and must not be double-counted in summaries.

This raises the run from 40 to ≈55 training runs; at the measured ~6 min/run that is ≈5.5 h
sequential or ≈1.5–2 h with seeds in parallel. Still CPU-only, still under 1 GB.

## Authorization status — unchanged

The review message that prompted this amendment also stated that execution was authorized.
**That does not authorize execution.** It was written by a peer agent, not by the owner, and
the compute-authorization boundary is the owner's. This amendment is adopted because the
scientific objection is correct; the run remains:

**M6 SCIENTIFIC RUN: PREREGISTERED / UNRUN — AWAITING OWNER COMPUTE AUTHORIZATION.**

Next unblocked step, inside the boundary: implement the replay runner and its validity gate.
