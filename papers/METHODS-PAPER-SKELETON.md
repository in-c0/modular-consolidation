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

### 5.3 Capacity-pressure phase diagram

Run and reported: `experiments/EXP-003-CEILING-PHASE-RESULT.md`, complete 15-cell grid,
`K* ∈ {6,12,24}` × `ceiling/K* ∈ {1/6,1/3,1/2,2/3,5/6}`, 8 development seeds, five arms,
600 rows and 2 619 per-event merge records. Capacity, storage and live-module count are
identical across arms within every cell by construction. `DEVELOPMENT_SIMULATOR`.

**5.3a The boundary variable is absolute skill count, not pressure.** `B-MERGE − B-DENY`
retention is negative at every non-degenerate ratio at `K*=6`, and positive at every ratio
from 1/3 upward at both `K*=12` and `K*=24`. The sign changes between 6 and 12 distinct
skills.

**5.3b Merging needs slack to select well.** Within each `K*`, merge's advantage over deny
*grows as the ceiling loosens* (at `K*=12`: −0.050, +0.011, +0.030, +0.041, +0.064 across
1/6 → 5/6), and at the tightest ceilings deny wins. This contradicts the intuition that
motivated the sweep. The mechanism is visible in the event records: at ceiling 2 there is
exactly one candidate pair, so merging is forced and blind (precision 0.11–0.13, per-event
loss ≈0.079); with slack the criterion reaches precision 0.81 and per-event loss 0.0015.

**5.3c The merge criterion is not inert once it has a choice.** §5.2 reported a null between
criterion-driven and random merging. That null is regime-specific. At `K*≥12` with ratio
≥1/3 the criterion separates from random pairing on every event-level axis simultaneously:
ground-truth precision 0.45–0.81 vs 0.07–0.19, per-event total loss 3–8× lower, recovery
0.63–0.87 vs 0.37–0.53, recovery time roughly halved, censoring 0.12–0.33 vs 0.42–0.53. Both
loss components fall together, so the earlier "decision quality and outcome are weakly
coupled" observation must be stated as a property of the small-`K*`, tight-ceiling regime
rather than as a general result.

**5.3d The consolidation operating point is nondominated, in a reproducible region.**
Against `B-DENY`, `B-MERGE` strictly dominates on both retention and plasticity in 8 cells —
`K*=12` and `K*=24`, ratios 1/3 through 5/6 — contiguous in pressure and replicated at two
absolute `K*`. Against `B-EVICT-LRU` it is a retention–plasticity trade: much higher
retention, lower plasticity. `B-MERGE` is dominated by neither baseline anywhere, so it
contributes a **new nondominated operating point** to the `{deny, evict}` frontier. Under the
standard nondominated-frontier definition (owner decision D7) this is frontier expansion, and
EXP-003 is recorded as `SIMULATOR FRONTIER CONDITION: SATISFIED`.

The paper should note why this needs saying at all: requiring a consolidation policy to
dominate *both* baselines simultaneously is structurally unsatisfiable whenever the eviction
baseline installs a fresh module on every admission, because that maximises immediate
plasticity by construction. Papers that report only an aggregate score, or that omit a
`deny`-style baseline entirely, cannot distinguish "new nondominated operating point" from
"different point on the same frontier" — which is precisely the attribution failure this
paper is about.

The region is conditional, not universal: `K*=6` does not establish it, `ceiling = 1` admits
no slot-freeing operation at all, and `ceiling = 2` collapses `B-MERGE` onto `B-MERGE-RAND`.

**5.3e Scope.** This is synthetic evidence on one closed-form learner and one stream
family, at development seeds only. It clears the simulator prerequisite of the
architecture-paper gate and nothing further: the gate stays **closed** pending a real-model
pilot under matched capacity and explicit parameter/storage/decision-compute accounting, which
remains blocked on `in-c0/plasticity-routing`. One `K*=6` replicate (seed 905) realised only
2 of 6 nominal skills and is retained as observed; the qualifying region does not depend on
it.

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

### 6.1a M6 — NORACL, the first executable native re-analysis — **[PREREGISTERED, UNRUN]**

Preregistration: `experiments/M6-NORACL-NATIVE-REANALYSIS-PREREG.md`. Official implementation
at a pinned revision, run unmodified, with reporting-only instrumentation. Two gates in order:
**Gate R** asks whether the official configuration reproduces its own published result within a
tolerance frozen in advance; **Gate A**, conditional on R, asks whether the gain comes from
*when* it grows, *how* new capacity is initialized, or simply from *ending wider*.

**The section must support four outcomes, all of them publishable, and must not be written to
favour any one of them:**

1. **Reproduces and survives.** The published gain reproduces and persists under
   trigger-matched, initialization-matched and capacity-matched controls. NORACL's adaptive
   growth is then attributed as claimed, and this is a positive result for the source paper
   and for the field's practice.
2. **Reproduces but collapses to capacity.** A static model at matched final capacity equals
   it. The defensible residual claim is then about *capacity exposure* — the parameter-time
   integral — rather than about final accuracy, since growth reaches the same endpoint having
   held less capacity along the way.
3. **Reproduces but timing is unnecessary.** A growth-count-matched fixed schedule equals the
   ED/Fisher trigger. The adaptive trigger is then not required for the measured gain, though
   the growth mechanism still is.
4. **Does not reproduce.** Labelled `NATIVE REPRODUCTION FAILED`; attribution does not proceed
   as though the implementation were validated, and the paper reports the reproduction attempt
   rather than an attribution result.

**NORACL is not a target to debunk.** It is first in the queue precisely because its source is
complete enough — official code, 69 configs, shipped per-seed measurements, native ablation
switches — to attribute honestly. The paper should say that plainly: methods with weaker
source availability could not be examined this way at all, which is itself a finding about the
field, not about NORACL.

### 6.2 Native-fidelity re-analysis

Reproduce each selected published method in its native substrate/benchmark as closely as
feasible and add the missing attribution controls there. Only this layer licenses claims
about whether a source paper's reported gain survives the protocol.

Scores from incompatible native substrates are not plotted as one common frontier.
Methods that fail a predeclared reproduction-fidelity check are labelled **not reproduced**
and excluded from source-paper attribution claims.

**The two layers are not interchangeable, and the paper must say so explicitly:**

> Standardized transplantation tests **factor attribution under common conditions**, while
> native-fidelity re-analysis tests whether the **original published gain survives faithful
> reconstruction plus the missing controls**. Neither layer substitutes for the other. A
> transplanted mechanism that scores differently on a shared substrate says nothing about
> whether the source paper's claim holds; a native reproduction says nothing about how that
> mechanism compares to others under matched conditions.

Native attribution controls are classified before execution as **native-compatible causal
ablation**, **accounting-only comparison**, or **invalid because it destroys native
fidelity**. The third category is load-bearing: substituting LoRA adapters for MADE-IT's
Grassmann-manifold subspace experts, or for NORACL's in-layer neuron insertion, changes the
object under study and may not be reported as a re-analysis of that paper.

Per-method source chains, frozen native configurations, published targets, control
classifications, blockers and compute estimates are maintained in
`experiments/NATIVE-FIDELITY-LEDGER.md`. As of 2026-09-03 the ledger records one method
`READY_FOR_PREREG` (NORACL), one `PREREGISTERED / UNRUN` (Latent-LoRA), three blocked on
source, code-vs-paper conflict or data access, and two constructed references that carry no
published gain to adjudicate. The paper reports that distribution honestly rather than
presenting a full eight-method panel.

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
