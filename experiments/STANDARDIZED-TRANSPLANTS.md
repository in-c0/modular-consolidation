# Standardized mechanism transplant contracts

**Status: DESIGN CONTRACT — no real-model run has been executed.**

This document resolves owner decision D6 as far as the source papers permit before the
shared model/rank/routing substrate arrives from `in-c0/plasticity-routing`. Its purpose is
to prevent semantic drift: a method is not allowed into the Long Sequence / MTL15 panel
merely because some vaguely similar operation can be implemented with LoRA.

## Common Layer-A contract

The standardized panel uses the two Long Sequence task orders used by recent task-agnostic
LoRA continual-learning work (Orders 3 and 4 in Latent-LoRA):

- Order 3: `MNLI -> CB -> WiC -> COPA -> QQP -> BoolQ -> RTE -> IMDB -> Yelp -> Amazon -> SST-2 -> DBpedia -> AG News -> MultiRC -> Yahoo`
- Order 4: `Yelp -> Amazon -> MNLI -> CB -> COPA -> QQP -> RTE -> IMDB -> SST-2 -> DBpedia -> AG News -> Yahoo -> MultiRC -> BoolQ -> WiC`

All 15 tasks are classification tasks scored by accuracy. Inference is task-agnostic. Task
identity is retained only for scoring and the explicit `C-OID` oracle control. The canonical
task orders are not made recurrent or reordered after results are seen.

The following remain shared [BLOCKED] quantities until `plasticity-routing` resolves them:

- base language model;
- LoRA/adapter family and rank;
- common train/evaluation budget;
- common task-agnostic routing substrate where a transplant does not define its own router;
- exact handling of label/head selection for methods whose source setting assumes task IDs.

Every standardized run must count router storage and `decision_flops`. A transplant that
changes the source mechanism qualitatively is given a new descriptive name and is never
reported as the source method.

## Portability classes

- **P1 — direct mechanism:** the source already operates on a compatible PEFT/LoRA substrate
  or defines a substrate-independent rule that can be applied without changing its causal
  meaning.
- **P2 — explicit transplant:** the source rule can be preserved, but another component
  (expert family, task-ID handling, etc.) changes. It is useful for mechanism comparison but
  is not a reproduction.
- **P3 — semantic transplant:** moving the rule to LoRA changes what the operation means.
  It is excluded from the primary standardized panel unless independently validated.

## M1 — fixed bank + learned task-free router

**Class: constructed reference, not a paper reproduction.**

Freeze a bank of `K` identically parameterized adapters before the sequence. No allocation,
merge, eviction, retirement or compression is permitted. The shared task-free router chooses
or mixes adapters. `K` is derived from the target arm being controlled where the lattice
requires `C-TERM`/matched capacity; it is never selected from outcome scores.

Purpose: isolate routing from allocation/consolidation under identical adapter capacity.

## M2 — boundary-driven expansion reference

**Class: P2; standardized name `BEXP-LoRA`, not “DEN”.**

Progressive/DEN-family work establishes dynamic capacity expansion, but native DEN grows and
splits neurons with selective retraining rather than allocating one LoRA expert. Therefore
the common-panel mechanism is deliberately narrower:

1. on each training task boundary, allocate one fresh adapter;
2. freeze previously completed adapters;
3. do not merge, prune, evict or compress;
4. at task-agnostic inference, use the shared router;
5. expose source-like task-indexed selection only as `C-OID`.

This arm tests **boundary-driven unbounded expansion** under the same substrate. It does not
license a claim that DEN itself was reproduced.

Primary controls: `C-TERM(BEXP-LoRA)` and `C-OID(BEXP-LoRA)`. A random-timing spawn control
is not interpreted as a test of DEN's native criterion because this standardized arm has no
learned spawn criterion.

## M3 — MoCL-P composition + pruning

**Class: P2, provisionally admissible.**

Source: Wang et al., *Learn it or Leave it* (arXiv:2406.18708).

The source uses prefix tuning but explicitly states that adapters and LoRA can in principle
be combined with MoCL-P; that LoRA version was not evaluated in the paper. Therefore this is
an explicit PEFT-family transplant, not a source reproduction.

Mechanism to preserve:

1. allocate a fresh task-specific PEFT module for each training task;
2. freeze older modules;
3. learn a task feature vector in the same representation dimension used for module
   matching;
4. compute input-to-task matching weights from similarity between the input representation
   and learned task feature vectors;
5. train the current module while composing it with weighted prior modules;
6. after the task, apply the source pruning decision to the newly allocated module.

For MTL15, freeze the published pruning threshold `alpha_ths = 0.25`. The source selected
benchmark-specific thresholds after a threshold study; **we do not retune that threshold on
our standardized panel**. Any alternative threshold is a separately labelled sensitivity
analysis and cannot replace the frozen run.

### M3 blockers before execution

The paper states the pruning decision in terms of the newly learned module's matching weight
`alpha_m`, but the exact dataset-level aggregation used to turn per-input matching weights
into the final pruning statistic must be verified from an authoritative implementation or
source before code is frozen. Do not invent mean/max aggregation.

The source experiment is task-incremental and assumes task labels at training and testing.
D6 requires task-agnostic inference, so the task/head-selection adaptation must be fixed with
`plasticity-routing`; the source-like task-ID path is retained only as `C-OID`.

Until both points are resolved, M3 is **specified but not executable**.

Primary controls once executable: no-prune MoCL counterpart, `C-TERM`, `C-SHRINK`, and
`C-OID`.

## M4 — Zero-Leakage reconstruction novelty routing

**Class: P2, provisionally admissible.**

Source: Kermiche, *Modular Continual Learning via Zero-Leakage Reconstruction Routing and
Autonomous Task Discovery* (arXiv:2604.14375).

The portable causal mechanism is its **autonomous novelty/return decision**, not its whole
Teacher/Student pipeline. Preserve:

1. maintain a per-module reconstruction router over a frozen representation;
2. before adapting to a batch, compute familiarity as the minimum reconstruction error over
   existing routers;
3. use a strict hard novelty gate during allocation;
4. calibrate router `j`'s novelty threshold as
   `tau_j = mu_cal_j + max(3 * sigma_cal_j, m)`, where calibration statistics come from that
   router's acquisition holdout;
5. create a provisional expert/router only when the input is outside the existing familiar
   manifolds;
6. commit only after the provisional router has stable reconstruction and the expert meets
   the sustained performance criterion; otherwise it must not permanently pollute the bank;
7. use reconstruction routing for returning manifolds rather than spawning duplicates.

### M4 blockers before execution

The source's TB-AE architecture, minimum-margin `m`, commitment-window `K`, stability
criterion and soft inference rule must be fixed from source defaults or an independently
predeclared implementation. The standardized expert may be LoRA, but that substitution is
explicitly labelled; the novelty rule itself must not be replaced by a convenient learned
gate.

Primary controls: `C-TERM`, matched random routing, and `C-RSPAWN` derived from realised
spawn count/timing rules as defined by the lattice.

## M5 — MADE-IT

**Class: P3 for Layer A; native-fidelity only for now.**

Its source mechanism is continual model merging on a CLIP/Vision Transformer substrate.
Mapping the entire merge rule directly onto a Long Sequence LoRA bank changes both object
geometry and routing semantics. Keep M5 in Layer B unless a separate validation demonstrates
that a specific LoRA transplant preserves the source merge criterion's meaning.

## M6 — NORACL

**Class: P3 for Layer A; native-fidelity only for now.**

Source: Raghunathan et al., *NORACL: Neurogenesis for Oracle-free Resource-Adaptive
Continual Learning* (arXiv:2604.27031).

The previous matrix entry “saturation-triggered LoRA growth” was too loose. Native NORACL:

- monitors normalized activation Effective Dimension for representational saturation;
- independently monitors current-task Fisher magnitude against an accumulated historical
  Fisher baseline for plasticity saturation;
- grows a layer only when **both** gates fire;
- adds a computed number of **neurons at the saturated layer**;
- initializes new fan-in directions orthogonally and fan-out weights at zero, making the
  insertion function-preserving;
- pads Fisher/EWC anchors with zeros so the new parameters begin unconstrained.

Replacing this with “spawn a fresh LoRA expert when the two signals fire” preserves the
*trigger* but not neurogenesis: locality, amount of growth, function-preserving insertion and
fresh-parameter geometry all change. Such an experiment may later be useful as an
exploratory `NORACL-trigger -> LoRA` arm, but it cannot enter the primary standardized panel
under the NORACL name without independent validation.

Layer-B native re-analysis remains valid and should add `C-RSPAWN`/matched-capacity controls
there.

## M7 — Latent-LoRA

**Class: P1/P2 boundary; primary standardized candidate.**

Source: Rahimi Azghan et al., *Latent-LoRA* (arXiv:2607.23837).

This method is already task-agnostic and evaluated on Long Sequence, so its routing mechanism
is directly relevant. Preserve:

1. one frozen adapter snapshot per training task;
2. stationary representations from the frozen base embedding layer using mean-pooled token
   embeddings;
3. a task-specific Gaussian mixture fit after each task;
4. posterior `p(task | x)` from the collection of GMMs at inference;
5. soft adapter blending under that posterior rather than task-ID selection.

The source adapter is a compact SVD-latent LoRA-XS-style `r x r` parameterization rather than
ordinary LoRA. If the common substrate uses ordinary LoRA, report two factors separately:
`M7-router` on the common adapter family and native Latent-LoRA in Layer B. Do not attribute
adapter-geometry effects to the GMM router.

Primary controls: identical adapter bank with random/matched routing, `C-OID`, and the
terminal-capacity control required by the lattice.

## M8 — FLAME

**Class: P3 for Layer A; native-fidelity only for now.**

Its native continual multimodal MoE/compression substrate is not semantically equivalent to
a Long Sequence LoRA bank. It stays in Layer B unless a separately validated standardized
compression rule is defined.

## Layer-A admission state after source verification

| Arm | Status before `plasticity-routing` | Reason |
| --- | --- | --- |
| M1 | blocked only on shared substrate | constructed reference |
| M2 `BEXP-LoRA` | blocked only on shared substrate/router | deliberately constructed P2 reference |
| M3 MoCL-P transplant | **blocked** | pruning-stat aggregation + task-agnostic adaptation unresolved |
| M4 reconstruction novelty transplant | **blocked** | router/threshold implementation constants unresolved |
| M5 MADE-IT | **excluded for now** | P3 semantic transplant |
| M6 NORACL | **excluded for now** | neuron neurogenesis != LoRA expert spawn |
| M7 Latent-LoRA/router | blocked on shared adapter choice | native benchmark/router already compatible |
| M8 FLAME | **excluded for now** | P3 semantic transplant |

Thus the standardized panel is not “six methods waiting only for a base model.” The clean
starting core is M1 + M2 + M7, with M3/M4 admitted only after their explicit blockers are
closed. This shrinkage is intentional: mechanism fidelity is more important than filling the
matrix.

## Source anchors

- MoCL-P: https://arxiv.org/abs/2406.18708
- Zero-Leakage Reconstruction Routing: https://arxiv.org/abs/2604.14375
- NORACL: https://arxiv.org/abs/2604.27031
- Latent-LoRA: https://arxiv.org/abs/2607.23837

These anchors are design provenance, not experimental evidence from this repository.
