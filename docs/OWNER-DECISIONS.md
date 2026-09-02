# Owner decisions log

Decisions that are the owner's to make, recorded with their date and consequences. This
file is append-only; superseded decisions are struck through, not deleted.

## 2026-09-02 — three §9 items from issue #1

### D1 — EXP-100 substrate: small LM + LoRA/expert banks

The intended eventual confirmatory substrate is a small language model with LoRA-style
modular adapters, **not** vision/classification.

Rationale (owner): the track belongs to the CCS continual-agent programme, so the substrate
must stay compositionally compatible with State Promotion and Plasticity Routing for later
integration. Vision may be used as a cheap diagnostic or benchmark side-study where it
isolates a methodological question better, but must not silently become the main scientific
claim.

Consequence: `experiments/EXP-100-PREREG-DRAFT.md` §7 records the substrate. Model family,
adapter rank and routing implementation remain **unfrozen**; the §6 dependency on
`in-c0/plasticity-routing` is still binding.

### D2 — scaled likely-negative EXP-100: not yet

Do not spend meaningful compute on the scaled experiment merely to test the current
negative-looking hypothesis. Revisit only after all three of:

1. `plasticity-routing` supplies the substrate/routing definitions required by §6;
2. the interference regime demonstrates the benchmark can distinguish consolidation from
   capacity;
3. the resulting design would answer a question not already settled by cheaper experiments.

This is a **sequencing** decision, not avoidance of falsification. A valid negative scaled
result remains publishable and desirable once the gates pass.

### D3 — primary paper: the decomposition/control protocol

N1 (budget-matched six-factor decomposition), N2 (event-level merge decomposition) and
N5 (randomised consolidation controls) become the primary near-term contribution: a
methods/evaluation paper about **attribution** in modular continual learning.

A new policy is explicitly *not* required for that paper to be worthwhile. Re-analysis or
reimplementation of representative published modular CL methods under the control lattice
is acceptable and may be scientifically stronger than proposing another spawn/merge
heuristic.

The architecture/EXP-100 paper becomes a **second, conditional** paper, written only if
real-model experiments reveal a nontrivial consolidation effect that survives the controls.

Consequence: see `papers/METHODS-PAPER-SKELETON.md` and `experiments/METHODS-MATRIX.md`.

## 2026-09-02 — second pass after EXP-001/EXP-002

### D4 — architecture paper remains conditional; do not pursue it now

`B-MERGE ≈ B-DENY` lowers the case for a separate architecture paper on current evidence.
The current binding-ceiling result belongs in the methods paper: merging is substantially
safer than destructive eviction, but at `K*=6, ceiling=3` it does not beat the strongest
missing baseline (`deny`) on retention; it exchanges some stability for plasticity.

Do **not** run EXP-100 to rescue an architecture narrative. Re-open the architecture paper
only if both stages below hold:

1. the simulator phase diagram shows a **reproducible region**, not one selected ceiling,
   where consolidation expands the retention–plasticity Pareto frontier relative to both
   deny and evict under identical binding capacity; and
2. a real-model pilot reproduces that frontier expansion while surviving parameter,
   storage and routing-decision-compute accounting.

If that regime never appears, no architecture paper is required. The methods paper remains
the primary contribution.

### D5 — run the ceiling-to-`K*` phase diagram now

This is the highest-value cheap simulator experiment. It tests whether the `B-DENY` result
from EXP-002 is specific to small `K*` plus recurrence.

Predeclare the sweep before running any grid cell. Vary absolute skill count and pressure
separately:

- `K* ∈ {6, 12, 24}`;
- `ceiling / K* ∈ {1/6, 1/3, 1/2, 2/3, 5/6}` (all integer for these `K*` values);
- arms: `B-DENY`, `B-MERGE`, `B-EVICT-LRU`, `B-EVICT-RAND`, `B-MERGE-RAND`;
- hold expected exposures per skill and recurrence statistics fixed as `K*` grows by scaling
  stream length with `K*`; do not hold segment count fixed;
- development seeds only; report the complete grid, never a selected best ratio.

Primary paired comparison is `MERGE − DENY`, with retention and plasticity reported
**separately**. Also report `MERGE − EVICT-LRU` across the complete grid. Forgetting, merge
precision, per-event loss and recovery are explanatory outcomes.

Interpretation is fixed before the run:

- if merge only trades retention for plasticity versus deny, it remains a methods-paper
  operating-point result;
- if merge yields a reproducible region that extends the deny/evict Pareto frontier as
  pressure rises, that is the first simulator-level reason to re-open the architecture paper;
- a benefit at one isolated ratio remains exploratory until reproduced at a second absolute
  `K*`.

Do not tune recurrence after seeing the phase diagram. Any recurrence sweep is a later,
separately predeclared experiment.

### D6 — methods-paper benchmark: Long Sequence / MTL15, with two evidence layers

~~A single native-fidelity benchmark across all eight selected methods is required for the
frontier plots.~~ That requirement is withdrawn because it conflates two different goals:
cross-method mechanism comparability and faithful attribution of a published result.

Use two evidence layers:

**A. Standardized mechanism panel.** Use **Long Sequence / MTL15** as the common small-LM +
adapter benchmark. Use the two established fixed task orders, task-agnostic inference, and
task identity only for scoring and `C-OID`. Prefer this over SuperNI for the causal panel
because the 15 tasks share classification accuracy, so the retention/plasticity frontier has
a common metric. Implementations transplanted onto this substrate must be labelled
**standardized mechanism implementations**, not reproductions of the source paper's
headline result. Model family, LoRA rank and routing implementation remain gated on
`plasticity-routing`.

**B. Native-fidelity re-analysis.** Reproduce each selected method on its native substrate
and benchmark as closely as feasible, then add the missing attribution controls there.
Claims about whether a published gain survives the control lattice come only from this
layer. MADE-IT (CLIP-ViT continual model merging) and FLAME (continual multimodal multi-task
MoE) stay in this layer unless a LoRA transplant is independently validated.

A recurrent-return version of Long Sequence / MTL15 may be added only as a separately
named, predeclared stress test; the canonical sequence must not be silently modified.

Consequence: the benchmark is no longer the conceptual blocker. Remaining blockers are
(1) exact standardized transplant definitions for methods whose native substrate differs,
and (2) the model/rank/routing substrate dependency on `in-c0/plasticity-routing`.

### Standing instruction

Preserve EXP-000 exactly as a negative development result. Do not tune it away.
