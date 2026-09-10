# M6 Gate R — environment diagnosis, and the root cause

**`M6 NATIVE REPRODUCTION: FAILED` stands.** The environment hypothesis is **refuted**, and
the actual cause is a source-availability gap.

## The environment hypothesis is refuted

The first Gate R ran on JAX 0.11.1. The source declares `jax>=0.4.28`, so I built a
second environment inside that declared range — Python 3.12, **JAX 0.4.30**, optax 0.2.3,
numpy<2 — and re-ran all five seeds into this separate directory. The original results were
committed unmodified first (`8e1ed77`) and are untouched.

| seed | published | JAX 0.11.1 | JAX 0.4.30 | params (0.4.30) | w0 | w1 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 75.17 | 82.80 | 77.33 | 9 576 | 12 | 12 |
| 1 | 73.53 | 67.82 | 71.57 | 9 576 | 12 | 12 |
| 2 | 71.26 | 82.23 | 74.58 | 9 632 | 12 | 16 |
| 3 | 71.95 | 83.19 | 82.39 | 9 660 | 12 | 18 |
| 4 | 74.27 | 74.79 | 82.29 | 9 576 | 12 | 12 |

**Mean parameters: 9 604 (JAX 0.4.30) vs 9 601 (JAX 0.11.1) vs 19 255 published.**

Two JAX generations, four years apart, agree with each other to within 3 parameters and
disagree with the published value by **−50.1%**. Growth under-fires identically in both.
All four Gate-R criteria fail again. **JAX version is not the cause.**

## Root cause: the producing configuration is not shipped

Comparing what the repository ships for each arm:

- `results/paper/bsmnist_2l_static32_s0/config.json` — a **complete 33-field config**
  including `config_name: bsmnist_2l_static32`, `hidden_dim`, `gamma`, `f_sat_percentile`,
  `orth_thresh`, `qr_init_scale` and the rest. Accompanied by `per_epoch.csv`,
  `acc_matrix.csv`, `summary.json`, `timing.csv`. These are genuine **run artefacts**.
- `results/paper/bsmnist_2l_noracl_s0/config.json` — an **8-field stub**: dataset, n_tasks,
  n_out, importance, run_name, n_layers, grow, and
  `"source": "published values (see PROVENANCE.md)"`. Accompanied only by `per_task.csv`.

**The stub pins none of the growth hyperparameters** — no `gamma`, no `f_sat_percentile`, no
`orth_thresh`, no `hidden_dim`, no `k_fixed`. And **`PROVENANCE.md` does not exist anywhere in
the repository**; the file the stub points to for provenance is absent.

So the NORACL rows are **transcribed published values, not run outputs**, and the
configuration that produced them is not in the repository. The static rows are real run
outputs; the NORACL rows are not.

This retracts an assumption in the preregistration. I selected the shipped per-seed
measurements as the Gate-R target precisely because they looked like a **traceable
config↔result pair**, in contrast to a table cell. They are not traceable: the config half is
a stub and its provenance pointer is dangling.

## Three mutually inconsistent parameter counts

| source | mean final parameters | status |
| --- | --- | --- |
| paper Table 1 (2L, Binary Split MNIST) | ≈23 300 | not reproducible from shipped code |
| repo `results/paper/*_noracl_*` | 19 255 | transcribed values, producing config not shipped |
| shipped config run unmodified, two JAX generations | **9 601 / 9 604** | reproducible, and stable |

The only one of the three that is reproducible is the one nobody published.

## What this is, and what it is not

**It is** a finding about source availability and internal consistency: the official
`bsmnist_2l_noracl.yaml`, run unmodified at the pinned revision, does not produce the
published NORACL capacity figures, and the configuration that would is not distributed.

**It is not** a claim that NORACL's mechanism is wrong, that its published result is wrong, or
that the authors did anything improper. A missing provenance file and a stubbed config are
ordinary release oversights. The mechanism itself was validated as sound in the earlier
score-free smoke: function-preserving insertion measured exactly `max|Δoutput| = 0.000e+00`.

**It is exactly what Layer B exists to detect.** Attribution controls would have been run on
top of an implementation that does not reproduce its own published numbers, and every
downstream comparison would have inherited that.

## Consequences

1. **Attribution does not proceed.** No `T-*`, `I-*` or `C-*` arm has been run. Deriving
   `T-COUNT.k_fixed` and `C-MATCH.hidden_dim` from a `T-FULL` that ends at half the published
   capacity would propagate the defect into every control.
2. **No configuration will be searched for to close this gap.** Hunting for settings that
   reproduce 19 255 parameters would be outcome-guided tuning — forbidden by the
   preregistration and by the standing rule against choosing between implementations by which
   reproduces the reported number.
3. **The one legitimate unblock is an author clarification**: the actual
   `bsmnist_2l_noracl` configuration used for the published runs, or the missing
   `PROVENANCE.md`. That is an owner-facing action (contacting authors), not something to
   resolve unilaterally.
4. **M6's ledger status changes** from `PREREGISTERED / UNRUN` to
   **`NATIVE REPRODUCTION FAILED — BLOCKED_SOURCE_AMBIGUITY`**, joining M3 and M4. The
   irony is recorded honestly: M6 was promoted to first-in-queue *because* it had official
   code, and official code turned out not to include the configuration that produced the
   published result.

## Artefacts

Immutable. `results/m6_gate_r/` (JAX 0.11.1) and `results/m6_gate_r_env_jax0430/`
(JAX 0.4.30) are both preserved; neither overwrites the other.
