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

### Standing instruction

Preserve EXP-000 exactly as a negative development result. Do not tune it away.
