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
method; no matched-budget decomposition separates all six. So the narrow question is:

> **Does consolidation buy anything that capacity cannot?**
>
> In a task-free stream, at matched stored parameters, active parameters, cold-storage
> bytes and total algorithmic compute (routing decisions included), does a policy that
> spawns, merges, retires and reinstates modules beat a fixed bank of the same terminal
> size with the same learned routing?

The sharp version: **is transient over-allocation followed by consolidation better than
being right-sized from the start?** That is the only thing consolidation can uniquely buy,
because you do not know the right size in advance.

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

## Development pilot: the falsification already fires

CAMS-v0 is a synthetic stream with a **known ground-truth skill count `K*`**, so
over-allocation, under-allocation and merge correctness are measurable directly rather than
inferred from accuracy. On it, over 5 development seeds:

- learned routing beats random routing at identical capacity (0.809 vs 0.630 retention) —
  and a bank with a bad router is worse than no bank at all (0.630 vs 0.681);
- a fixed bank of the full lifecycle's own final size beat the full lifecycle by 7.8 points
  at identical capacity (0.738 vs 0.660);
- the merge criterion picked ground-truth-correct pairs with precision 1.00 versus 0.67 for
  random pairing — and it barely mattered, costing ~0.002 accuracy either way.

**On this toy, consolidation bought nothing that capacity could not.** Full write-up,
including the benchmark-validity failure found on the first configuration and the two
accounting bugs found by the tests, is in
[`experiments/EXP-000-TOY-RESULT.md`](experiments/EXP-000-TOY-RESULT.md).

This does not falsify the hypothesis for real models — the toy learner merges almost
losslessly and its retention tracks capacity almost linearly, both artefacts. It does raise
the burden of proof, and the confirmatory design is built against this null.

## Repository layout

```
docs/LITERATURE-AUDIT-2026-09-02.md   what already exists, and what is genuinely open
docs/ARMS.md                          the one-factor-at-a-time lattice and its controls
docs/METRICS.md                       metric definitions, including capacity and merge decomposition
docs/TERMINOLOGY.md                   fixed vocabulary shared with sibling tracks
docs/BENCHMARK-POLICY.md              how difficulty may be calibrated without cheating
docs/DEPENDENCIES.md                  contamination rules and evidence status
experiments/EXP-000-TOY-RESULT.md     development pilot, including its negative result
experiments/EXP-100-PREREG-DRAFT.md   candidate confirmatory protocol -- NOT FROZEN
src/modular_consolidation/            generic module lifecycle infrastructure
```

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

`make calibrate` chooses stream difficulty by a method-independent criterion on development
seeds. `make toy` runs the arm lattice and every derived control, and prints the frontier
table plus the merge-loss decomposition.

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
