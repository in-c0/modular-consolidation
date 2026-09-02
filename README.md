# Modular Consolidation

**Working research project:** whether a continual cognitive system can allocate, specialise,
merge, compress, retire and reuse modules without catastrophic interference *and* without
uncontrolled capacity growth.

> **Source of truth: [issue #1](https://github.com/in-c0/modular-consolidation/issues/1).**
> If a document here disagrees with it, that issue wins.

> Status: **research/design track.** No confirmatory experiment has been run and none may
> be frozen until the design lessons from `in-c0/plasticity-routing` are available. The
> only empirical content here is a development simulator whose headline finding is
> **negative**.

## The question

Every operation in the obvious architecture already exists. Dynamic experts are not novel;
neither is task-free discovery, expert merging, adapter banking, routing, or pruning. The
literature audit says so explicitly and names the prior work
([`docs/LITERATURE-AUDIT-2026-09-02.md`](docs/LITERATURE-AUDIT-2026-09-02.md)).

What is unresolved is attribution. A modular continual learner that beats a baseline may be
winning because of routing, capacity, compute, task-identity leakage, allocation timing, or
consolidation. Prior work ablates these one at a time inside papers advocating a specific
method; no matched-budget decomposition separates all six. So the question is:

> **Does consolidation buy anything that capacity cannot?**

The first version of that question was posed in the unbounded-capacity regime. **That
version turns out to be close to analytically settled, in the negative** — see the finding
below. The live question is the constrained one:

> Under a **hard capacity ceiling below the number of distinct skills**, and at identical
> live-module count, parameters, storage and compute, does a policy that frees a slot by
> **pooling** modules (merge) retain more than one that frees it by **destroying** a module
> (evict) or by **refusing** to admit a new one (deny)?

Capacity is equal by construction there, so any difference is attributable to the slot
decision alone. It is also the realistic deployment case: fixed memory, unbounded stream.

## Why the obvious ladder is not the experiment

The candidate hierarchy — single adapter → fixed bank → learned routing → dynamic spawning
→ spawn+merge → spawn+merge+retire — changes several factors per rung. A win at rung 4
would be uninterpretable. It is replaced by a **lattice** where each edge changes exactly
one factor, plus controls whose configuration is derived from the realised behaviour of
the arm they control for. See [`docs/ARMS.md`](docs/ARMS.md).

The single most important control is the one prior work almost never runs: take the
dynamic method's own final module count, train a *fixed* bank of exactly that size with
the same router and the same compute, and compare. Without it, "dynamic allocation helps"
is indistinguishable from "N modules is the right size."

## Capacity is a metric, not a footnote

A method that avoids forgetting by allocating unboundedly many parameters has not solved
anything. Every arm is scored on `param_total`, `param_active`, `param_peak`,
`storage_total` (**including cold storage for retired modules**) and
`total_algorithmic_flops` (**including routing decision compute**). Primary results are
reported as a retention–plasticity frontier against capacity and compute, not as a table of
accuracies. See [`docs/METRICS.md`](docs/METRICS.md).

## What the simulator has established

CAMS is a synthetic stream with a **known ground-truth skill count `K*`**, so
over-allocation, under-allocation and merge correctness are measured rather than inferred.
All results below are **development-simulator** results on a closed-form ridge learner, not
evidence about real models.

### 1. Over-allocation is free in accuracy terms — so consolidation cannot fix forgetting

The retention-versus-capacity curve is **monotone non-decreasing** under any competent
routing. A module that is never selected cannot damage a prediction, so spare modules cost
parameters, compute and storage but not accuracy. Established across three independent
mechanisms — interference between skills, data scarcity per module, and soft density-gated
routing — none of which could produce an interior optimum
([`EXP-001`](experiments/EXP-001-INTERFERENCE-RESULT.md)).

**Consequence: at matched capacity in the unbounded regime, consolidation has nothing to
fix.** Its only possible benefit is moving left along the efficiency frontier — a
*compression* claim, not a *forgetting* claim. This explains
[`EXP-000`](experiments/EXP-000-TOY-RESULT.md), where a fixed bank of the full lifecycle's
own final size beat the lifecycle by 7.8 points at identical capacity.

### 2. Under a binding ceiling, merging and eviction are not the same operation

At an identical ceiling of 3 modules with `K* = 6` — same live-module count, same 576
parameters, same 33 408 bytes of storage for every arm — over 8 paired seeds
([`EXP-002`](experiments/EXP-002-CEILING-RESULT.md)):

| freeing a slot by | retention | plasticity | forgetting |
| --- | --- | --- | --- |
| refusing to spawn (`B-DENY`) | **0.762** | 0.775 | **0.032** |
| pooling two modules (`B-MERGE`) | 0.745 | 0.798 | 0.071 |
| deleting a module (`B-EVICT-LRU`) | 0.541 | **0.854** | 0.324 |

- **merge − evict: +0.204 retention**, 95% CI [+0.125, +0.296]. Pooling recovers essentially
  all of eviction's damage. This is the defensible form of the consolidation claim.
- **merge − deny: −0.017 retention**, CI spans zero — but that null conceals two significant
  effects in opposite directions: plasticity **+0.023** and forgetting **+0.039**, both CIs
  excluding zero. Consolidation converts stability into plasticity at roughly par.
- **merge − random-merge: +0.009**, CI spans zero. The criterion picks better pairs
  (precision 0.50 vs 0.36) and each merge is less damaging (0.027 vs 0.042), and none of it
  survives to the aggregate.

Two things follow for the field. `deny` is a missing baseline, and it is the strongest arm
on retention here. And "prune redundant experts" and "merge redundant experts" are routinely
used interchangeably while differing by 0.204 retention at identical capacity.

## Repository layout

```
docs/LITERATURE-AUDIT-2026-09-02.md   what already exists, and what is genuinely open
docs/ARMS.md                          the one-factor-at-a-time lattice and its controls
docs/METRICS.md                       metric definitions, including capacity and merge decomposition
docs/TERMINOLOGY.md                   fixed vocabulary shared with sibling tracks
docs/BENCHMARK-POLICY.md              how difficulty may be calibrated without cheating
docs/DEPENDENCIES.md                  contamination rules and evidence status
docs/OWNER-DECISIONS.md               decisions that are the owner's, with consequences
experiments/EXP-000-TOY-RESULT.md     first pilot: capacity explains everything
experiments/EXP-001-INTERFERENCE-RESULT.md  negative instrument result, and why it matters
experiments/EXP-002-CEILING-RESULT.md the binding-ceiling regime; the one positive result
experiments/METHODS-MATRIX.md         which published methods get re-analysed, and how
experiments/EXP-100-PREREG-DRAFT.md   candidate confirmatory protocol -- NOT FROZEN
papers/METHODS-PAPER-SKELETON.md      primary paper outline; no results may be written yet
src/modular_consolidation/            generic module lifecycle infrastructure
```

## Primary paper

Per owner decision D3, the primary near-term output is a **methods/evaluation paper about
attribution**, which proposes no new policy and re-analyses representative published modular
CL methods under the control lattice. See
[`papers/METHODS-PAPER-SKELETON.md`](papers/METHODS-PAPER-SKELETON.md) and
[`experiments/METHODS-MATRIX.md`](experiments/METHODS-MATRIX.md). The architecture paper is
a second, conditional paper.

Its thesis is falsifiable in the useful direction: if published gains **do** survive
capacity- and compute-matched controls, the field's evaluation practice is adequate and the
paper becomes a validation of it rather than a critique.

## Local gates

```bash
make test
```

```bash
make calibrate
```

```bash
make toy
```

```bash
make ceiling
```

`make calibrate` and `make calibrate-v1` choose stream difficulty by method-independent
criteria on development seeds. `make toy` runs the arm lattice and every derived control.
`make ceiling` runs the binding-ceiling regime with paired bootstrap CIs.

## Novelty boundary

Not claimed as novel: dynamic expert allocation, task-free expert discovery, expert
merging, adapter banks, routing over adapters, module pruning, saturation-triggered growth.

Claimed, if the experiments support it: the budget-matched six-factor decomposition and its
derived controls; event-level merge-loss decomposition into decision, mechanism and
interference components; a stream with ground-truth module structure; retirement with
reinstatement as distinct from pruning, with cold storage charged; and randomised
consolidation controls that separate the criterion from the rate.

The defensible contribution of this track is a decomposition and a control protocol, not an
architecture.

## Reproducibility policy

- Preregister hypotheses and invalidation criteria before confirmatory runs.
- Publish negative results; EXP-000's is in the README above, not buried.
- Keep development/calibration seeds disjoint from confirmatory seeds.
- Charge cold storage; count routing decision compute; match budgets before interpreting
  architecture effects.
- Never tune benchmark difficulty, thresholds or the success criterion toward a favoured
  arm; difficulty is calibrated by a criterion that mentions no candidate method.
- Ground-truth fields exist for scoring only; a policy that reads them makes the run invalid.

## License

Apache-2.0.
