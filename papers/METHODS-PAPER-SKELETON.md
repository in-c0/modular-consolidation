# Paper skeleton — attribution in modular continual learning

**Working title:** *Consolidation or Capacity? Attributing Gains in Modular Continual Learning*

**Status: skeleton. No real-model results section may be written until the experiments in
`experiments/METHODS-MATRIX.md` have been run.** Every section below states what would be
written, and sections depending on unrun experiments say so explicitly.

Per owner decisions D3–D6: this is the primary paper, it proposes no new policy, and it must
stand on the decomposition and controls alone. A separate architecture paper is conditional
on evidence that consolidation expands a retention–plasticity Pareto frontier under a
binding capacity ceiling; it is not a required deliverable.

---

## 1. Introduction

**Claim.** Modular continual learning reports gains that are often not cleanly attributed. A
method that beats a baseline may be winning from routing, capacity, compute, task-identity
leakage, allocation timing, or consolidation. Existing evaluations rarely separate all of
these at matched realised budgets.

**Contribution.** A control protocol, not an architecture:

1. a factor lattice in which controlled comparisons change one factor at a time;
2. controls derived from a target arm's *realised* behaviour, preventing post-hoc capacity,
   rate or event-count matching;
3. event-level merge-loss decomposition into decision, mechanism and interference terms,
   plus recovery-after-merge;
4. accounting that charges cold storage and routing decision compute;
5. a two-layer real-model evaluation: a common standardized mechanism panel and
   native-fidelity re-analyses of representative published methods.

**What we do not claim.** Dynamic allocation, expert routing, merging, pruning, compression
and retirement all predate this work. No new operation is required for the contribution.

## 2. Background and related work

From `docs/LITERATURE-AUDIT-2026-09-02.md`. Structure: expansion methods; parameter
isolation; task-free discovery; merging and continual model merging; pruning and
compression; capacity/accounting practice; HARC as direct prior art that merging can damage
routing as well as weights; CLEMC as the closest capacity-framing prior art requiring an
explicit distinction.

The section states plainly that each operation predates this paper.

## 3. The attribution problem

Factors: routing, capacity, compute, task identity, allocation, consolidation, slot policy
under a binding ceiling, and compression. Explain why a within-method ablation does not by
itself distinguish these factors.

Two regimes must be separated:

- **unbounded capacity:** consolidation can be a compression mechanism, but in a
  parameter-isolated system with competent routing spare modules cannot directly hurt
  accuracy;
- **binding capacity:** a method must choose what to do when a desired new module cannot be
  added, making `deny`, `evict` and `merge` distinct equal-capacity decisions.

## 4. Protocol

The lattice (`docs/ARMS.md`), derived controls, metrics (`docs/METRICS.md`) and validity
flags. Emphasise:

- routing decision compute grows with live module count and is usually uncounted;
- retirement is not deletion and cold storage must be charged;
- retention, plasticity and forgetting are reported separately;
- merge and prune/evict are not interchangeable operations;
- `deny` is a mandatory baseline under a binding ceiling.

## 5. Simulator structural results

These results are **development-simulator evidence only** and are labelled that way in every
section, including the abstract.

### 5.1 Over-allocation is free in accuracy terms under competent isolated routing

Across interference, data scarcity and soft density-gated routing in CAMS, the attempted K5
criterion could not be satisfied (`experiments/EXP-001-INTERFERENCE-RESULT.md`). A module
that is never selected cannot damage a prediction, so retention is monotone non-decreasing
with spare isolated capacity in this instrument.

**Consequence.** At matched capacity in the unbounded regime, consolidation cannot improve
retention by removing unused isolated modules. Any defensible benefit there is a compression
or compute/storage claim, not a forgetting claim, unless another interference mechanism is
present.

### 5.2 Under a binding ceiling, merging and eviction are different operations

At `K*=6, ceiling=3`, identical live-module count, parameters and storage, B-MERGE retained
0.204 more than LRU eviction (95% CI [0.125, 0.296]), while its retention difference from
B-DENY spanned zero (`experiments/EXP-002-CEILING-RESULT.md`). The latter comparison hid a
plasticity increase and a forgetting increase in opposite directions.

**Consequence.** “Prune redundant experts” and “merge redundant experts” are not equivalent
descriptions, and `deny` is a missing baseline in memory-pressure evaluations.

### 5.3 Capacity-pressure phase diagram — **[PREDECLARED, UNRUN]**

Per D5 and `experiments/EXP-003-CEILING-PHASE-PREREG.md`, sweep absolute `K*` and
`ceiling/K*` separately while holding expected exposures per skill and recurrence statistics
fixed. Report the complete grid. The preregistration predates the committed runner
`scripts/run_ceiling_phase.py`.

The primary question is not whether one cell favours merge. It is whether B-MERGE creates a
**reproducible region** that expands the retention–plasticity Pareto frontier relative to
both deny and evict at the same binding capacity. A one-ratio benefit is exploratory until
it recurs at a second absolute `K*`.

If no such region exists, the simulator evidence remains a methods/evaluation result and no
architecture narrative is inferred.

## 6. Real-model evidence design — **[REQUIRES UNRUN EXPERIMENTS]**

Per `experiments/METHODS-MATRIX.md`, use two evidence layers. Transplant fidelity and
admission are defined in `experiments/STANDARDIZED-TRANSPLANTS.md` before implementation.

### 6.1 Standardized mechanism panel — Long Sequence / MTL15

Use Long Sequence / MTL15 as the common small-LM + adapter benchmark, with the two
established task-agnostic Long Sequence orders, task-agnostic inference, and task identity
only for scoring and `C-OID`. This panel supports cross-method mechanism comparison under one
accounting protocol.

A mechanism transplanted here is labelled a **standardized implementation**, not a
reproduction of the source paper. After source-level fidelity review, the clean initial core
is narrower than the original matrix:

- M1 fixed-bank routing reference — candidate once the shared substrate is fixed;
- M2 `BEXP-LoRA` — explicit boundary-expansion reference, not a DEN reproduction;
- M7 Latent-LoRA/GMM routing — candidate, with adapter-geometry and router effects separated.

M3 MoCL-P and M4 reconstruction novelty routing are **blocked** until their exact transplant
semantics are closed. M5 MADE-IT, M6 NORACL and M8 FLAME are **native-fidelity only for now**:
forcing their CLIP-ViT merge, neuron-neurogenesis or multimodal-MoE operations into ordinary
LoRA would change the causal mechanism rather than standardize it.

Model family, adapter family/rank and the shared routing substrate remain gated on
`in-c0/plasticity-routing`.

### 6.2 Native-fidelity re-analysis

Reproduce each selected published method in its native substrate/benchmark as closely as
feasible and add the missing attribution controls there. Only this layer licenses claims
about whether a source paper's reported gain survives the protocol.

Scores from incompatible native substrates are not plotted as one common frontier.
Methods that fail a predeclared reproduction-fidelity check are labelled **not reproduced**
and excluded from source-paper attribution claims.

## 7. Results — **[MUST NOT BE WRITTEN YET]**

No real-model result may be described here before §6 is executed. Simulator results belong
in §5 and remain explicitly labelled simulator results.

When §6 runs, report:

- standardized-panel retention/plasticity vs capacity and vs total compute frontiers;
- native-fidelity control deltas per method, not a fake cross-substrate leaderboard;
- invalid runs and reproduction failures;
- event-level merge loss and recovery for methods that merge.

## 8. Limitations

- Simulator results use a closed-form toy learner where merge mechanism loss is unusually
  small; gradient-trained adapters may change the mechanism-loss and routing-loss terms.
- The EXP-003 pressure phase diagram is predeclared and has a committed runner, but no
  EXP-003 result exists until that runner is actually executed and its complete payload is
  committed.
- Reimplementation fidelity is the dominant threat to native-fidelity attribution claims.
- Standardization itself can change a method's causal mechanism; P3 semantic transplants are
  therefore excluded rather than treated as approximate reproductions.
- Long Sequence / MTL15 gives one common LM classification substrate, not universal coverage
  of multimodal or generative continual learning.
- Exact model family, adapter rank and shared routing implementation remain gated on
  `plasticity-routing`.

## 9. What would falsify or narrow the paper's thesis

If published gains consistently survive terminal-capacity-, compute-, storage- and
identity-matched native controls, the critique of current attribution practice is too strong;
that becomes the headline and the protocol functions as validation rather than indictment.

If standardized mechanisms occupy meaningfully different frontier regions while native
published gains survive their controls, the contribution narrows to a common measurement
framework rather than evidence of confounding.

The architecture paper is re-opened only if the simulator phase diagram **and** a real-model
pilot show consolidation expanding the retention–plasticity Pareto frontier relative to
deny and evict under the same binding capacity and accounting. EXP-100 is not run to rescue
a narrative.

---

## Writing rules for this repository

1. No sentence may describe an unrun experiment in the past tense.
2. Any number appearing here must be traceable to a committed results file.
3. Simulator results are labelled as simulator results everywhere, including the abstract.
4. Related work states that the component operations are not novel in the paper's own voice.
5. Standardized transplants are never described as reproductions of source-paper results.
6. One favourable pressure cell never licenses a regime-level or architecture claim.
7. A source method cannot enter Layer A under its original name if standardization changes
   the causal operation rather than only the substrate.
