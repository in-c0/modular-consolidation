# Paper skeleton — attribution in modular continual learning

**Working title:** *Consolidation or Capacity? Attributing Gains in Modular Continual Learning*

**Status: skeleton. No results section may be written until the re-analyses in
`experiments/METHODS-MATRIX.md` have been run.** Every section below states what would be
written, and the sections that depend on unrun experiments say so explicitly.

Per owner decision D3: this is the primary paper, it proposes no new policy, and it must
stand on the decomposition and controls alone.

---

## 1. Introduction

**Claim.** Modular continual learning reports gains that are not attributed. A method that
beats a baseline may be winning from routing, capacity, compute, task-identity leakage,
allocation timing, or consolidation. The field ablates these one at a time inside papers
advocating a specific method; no matched-budget decomposition separates them.

**Contribution.** A control protocol, not an architecture:

1. a six-factor lattice in which each arm differs from its neighbour in exactly one factor;
2. controls derived from a target arm's *realised* behaviour, so they cannot be chosen after
   the fact;
3. event-level merge-loss decomposition into decision, mechanism and interference parts;
4. capacity accounting that charges cold storage and routing decision compute;
5. a re-analysis of representative published methods under the protocol.

**What we do not claim.** No operation in this literature is novel and we propose none.

## 2. Background and related work

From `docs/LITERATURE-AUDIT-2026-09-02.md`. Structure: expansion methods; parameter
isolation; task-free discovery; merging and continual model merging; pruning and
compression; the capacity-reporting practice across the field. The related-work section
must state plainly that each operation predates this paper.

## 3. The attribution problem

Six factors (routing, capacity, compute, task identity, allocation, consolidation) and why
ablating one at a time within a method does not separate them. The terminal-capacity-matched
fixed bank as the control the field almost never runs.

## 4. Protocol

The lattice (`docs/ARMS.md`), the derived controls, the metrics (`docs/METRICS.md`), and the
validity flags. Emphasis on three things the field under-reports:

- routing decision compute grows with live module count and is usually uncounted;
- retirement is not deletion and cold storage must be charged;
- retention, plasticity and forgetting must be reported separately.

## 5. Two structural results

These do not require the re-analyses and are already established on the simulator.

**5.1 Over-allocation is free in accuracy terms.** In a parameter-isolated system with
competent routing, the retention-versus-capacity curve is monotone non-decreasing: a module
never selected cannot damage a prediction. Established across interference, data scarcity
and soft routing (`experiments/EXP-001-INTERFERENCE-RESULT.md`). **Consequence: at matched
capacity in the unbounded regime, consolidation cannot improve retention. Its only possible
benefit is compression.** Papers claiming merging reduces forgetting are therefore claiming
something that requires a binding constraint, an interfering routing scheme, or a different
mechanism — and should say which.

**5.2 Under a binding ceiling, merging and eviction are not the same operation.** At
identical live-module count, parameters and storage, merging retained 0.204 more than LRU
eviction (95% CI [0.125, 0.296]), while being statistically indistinguishable from simply
refusing to spawn — the latter decomposing into a significant plasticity gain (+0.023) paid
for by significant extra forgetting (+0.039)
(`experiments/EXP-002-CEILING-RESULT.md`). **Consequence: "prune redundant experts" and
"merge redundant experts" are not interchangeable descriptions**, and `deny` is a missing
baseline.

## 6. Re-analysis of published methods — **[REQUIRES UNRUN EXPERIMENTS]**

Per `experiments/METHODS-MATRIX.md`. This section is not written until those runs exist.
The section will report, per method: reproduction fidelity; performance under the terminal-
capacity-matched control; the frontier position; and, where applicable, the merge-loss
decomposition. Methods that do not reproduce are reported as not reproduced and excluded
from attribution claims.

## 7. Results — **[MUST NOT BE WRITTEN YET]**

No result may be described here before §6 is executed. The two structural results in §5 are
simulator results and must be labelled as such wherever they appear, including in the
abstract.

## 8. Limitations

- The structural results are established on a closed-form toy learner where merging is
  near-lossless; gradient-trained adapters are expected to shift the mechanism-loss term and
  possibly §5.2.
- Reimplementation fidelity is the dominant threat to §6 and is reported per method.
- `deny` benefits from a small skill count with recurrence; the ceiling-to-`K*` ratio is an
  unrun sweep.
- Single benchmark family so far; the substrate decision (small LM + LoRA) is made but not
  frozen, pending `in-c0/plasticity-routing`.

## 9. What would falsify this paper's thesis

If, across the re-analysed methods, reported gains **do** survive terminal-capacity-matched,
compute-matched controls, then the field's evaluation practice is adequate and the paper's
central claim is wrong. That outcome is reported as the headline, and the paper becomes a
validation of existing practice rather than a critique of it. The protocol is useful either
way; the thesis is not.

---

## Writing rules for this repository

1. No sentence in this file may describe an unrun experiment in the past tense.
2. Any number appearing here must be traceable to a committed results file.
3. Simulator results are labelled as simulator results in every section, including the
   abstract.
4. The related-work section states that the operations are not novel, in the paper's own
   voice, not only in a footnote.
