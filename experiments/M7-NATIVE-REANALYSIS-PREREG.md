# M7 / Latent-LoRA — native-fidelity re-analysis preregistration

**Status: PREDECLARED DESIGN — UNRUN. No compute is authorized by this file.**

This is the first Layer-B design under owner decision D6. It is a source-paper
**reproduction + attribution** experiment, not EXP-100 and not an architecture-paper run.
It does not lift any dependency on `in-c0/plasticity-routing`; the standardized Layer-A
panel remains separately gated.

Source: Rahimi Azghan et al., *Latent-LoRA: Compact Latent-Space Adapters with
Gradient-Free Routing for Continual Learning*, arXiv:2607.23837v1 (2026).

The source reports each experiment as the mean of three random seeds but does not publish the
seed values. That prevents bit-for-bit reproduction; this preregistration fixes replacement
reproduction seeds before any run.

## 1. Native target and scope

Use the source paper's **T5-Large (770M) + Long Sequence** setting only for the first native
re-analysis. Do not begin with T5-XL or Llama models merely because a result is easier to
match there.

Run both canonical Long Sequence orders:

- Order 3: `MNLI -> CB -> WiC -> COPA -> QQP -> BoolQ -> RTE -> IMDB -> Yelp -> Amazon -> SST-2 -> DBpedia -> AG News -> MultiRC -> Yahoo`
- Order 4: `Yelp -> Amazon -> MNLI -> CB -> COPA -> QQP -> RTE -> IMDB -> SST-2 -> DBpedia -> AG News -> Yahoo -> MultiRC -> BoolQ -> WiC`

All 15 tasks are classification tasks and use exact-match accuracy. Inference is task
agnostic and replay-free. Do not add recurrent returns, reorder tasks, or use task identity
except in the explicit oracle control.

## 2. Source configuration frozen before implementation

### Backbone and adapter

- base model: T5-Large, 770M parameters;
- frozen pretrained backbone;
- target modules: query and value projections of every attention layer;
- compact SVD-latent adapter as defined by the paper:
  `Delta W_t = U_r Sigma_r R_t V_r^T`;
- `U_r`, `Sigma_r`, `V_r` frozen from a rank-`r` truncated SVD of the pretrained target
  weight;
- only the per-task square matrix `R_t` is trainable;
- adapter rank `r = 32`;
- scaling `alpha = 16`;
- after each task, its adapter snapshot is frozen permanently.

### Training

- optimizer: AdamW;
- learning rate: `3e-4`;
- batch size: `8`;
- schedule: constant learning rate;
- epochs per task: `30`;
- Long Sequence orthogonal-regularization coefficient: `lambda = 0.02`;
- no replay from previous tasks.

The orthogonal penalty is the source `Sigma`-weighted latent-space penalty over the current
adapter against all prior frozen adapter snapshots. Do not substitute ordinary LoRA
orthogonality.

### GMM router

The executable router contract is already frozen in
`experiments/M7-LATENT-LORA-ROUTER-CONTRACT.md` and is incorporated here by reference:

- router representation: mean-pooled frozen input embeddings;
- `K = 5` Gaussian components per task;
- K-means initialization, 30 iterations;
- one covariance shared across all tasks/components;
- covariance regularization `epsilon = 0.01`;
- uniform task prior;
- soft posterior routing and posterior-weighted adapter blending;
- no gradient-trained router and no replay of prior raw data to refit it.

## 3. Reproduction seeds

Use exactly three development/reproduction seeds:

`930, 931, 932`

These replace the source's unpublished seed identities. They are **not confirmatory seeds**
and must not later be relabelled as such.

Use the same three seeds for every control below, paired within task order.

## 4. Source numbers used only for the fidelity gate

The paper reports the following T5-Large Long Sequence means over three unspecified seeds:

| Configuration | Order 3 AP | Order 3 FM | Order 4 AP | Order 4 FM |
| --- | ---: | ---: | ---: | ---: |
| Latent-LoRA, GMM soft routing | 79.95 | 0.57 | 79.87 | 0.73 |
| compact adapter + ortho, sum-at-inference (router ablation) | 70.12 | 3.03 | 73.19 | 3.97 |

AP and FM are expressed in percentage points in the source paper.

These numbers are **not targets for hyperparameter tuning**. The source configuration above
is frozen first; run it once under the three preregistered seeds and report the discrepancy.

## 5. Fidelity gate before source-paper attribution

Because the original per-seed variance and seed identities are unavailable, use a deliberately
simple predeclared tolerance rather than tuning until the table is matched.

For the main Latent-LoRA configuration, call the native implementation **reproduced for
attribution** only if, on **each** Long Sequence order:

- mean AP is within `2.0` percentage points of the published mean; and
- mean FM is within `2.0` percentage points of the published mean.

This tolerance is ours, not a source-paper claim. It is fixed before execution.

The sum-at-inference ablation is a **secondary fidelity diagnostic** using the same ±2.0 pp
bands, but failure of that secondary check does not silently change the main gate. Instead:

- main pass + ablation pass: strongest fidelity state;
- main pass + ablation fail: main result may enter attribution analysis, but router-mechanism
  conclusions carry an explicit ablation-fidelity warning;
- main fail: label **NOT REPRODUCED** and do not make claims about whether the source paper's
  reported gain survives our controls.

A failed reproduction is a result. Do not alter rank, learning rate, epochs, lambda, router
constants, task order, preprocessing, seeds or tolerance to rescue it. Any implementation
bug may be fixed only with a committed diagnosis that is independent of comparative outcome.

## 6. Native arms

Every arm uses the same T5-Large backbone, compact adapter geometry, training sequence,
optimizer budget and frozen task adapters unless the arm definition explicitly changes the
inference rule.

### N7-BASE — source Latent-LoRA

Source compact adapter + source GMM posterior + soft posterior-weighted adapter blending.
This is the reproduction arm.

### N7-SUM — source router ablation

Same trained compact adapters and orthogonal regularization, but at inference sum all stored
adapters as in the source ablation. No task router is used for selection.

Purpose: reproduce the source's own decomposition of compact adapter vs router.

### N7-RAND-MATCH — content-randomized matched router

Compute the source GMM posterior exactly as N7-BASE, then deterministically permute the
posterior weights across adapter identities for each evaluation example using a seed derived
only from `(reproduction_seed, order, task_index, example_index)`.

This preserves, per example:

- the posterior weight multiset;
- entropy/sparsity/softness;
- the number and shape of blended adapters;
- GMM likelihood computation and therefore decision-compute class;

while breaking the semantic mapping between router evidence and adapter identity.

Purpose: test whether **routing content** matters beyond paying for the same router and
producing the same mixture concentration. The permutation rule is fixed before results and
must not use labels or performance.

### N7-OID — oracle task identity

Use a one-hot posterior on the ground-truth task adapter at inference. Task IDs remain
unavailable to every non-oracle arm.

Purpose: price residual routing error/headroom. This is an oracle upper-bound control, not a
fair deployable comparator. Its lower decision compute is reported rather than artificially
padded.

### N7-TERM — terminal-capacity fixed-bank path control

Reserve all 15 compact-adapter slots before task 1, then train exactly one designated slot at
successive task boundaries using the same source optimizer/objective and freeze each slot
after its task. Use the same GMM router as N7-BASE.

Untrained future slots are never selected or mixed. Their parameter/storage reservation is
charged from the beginning.

Purpose: distinguish the *path of allocating capacity over time* from ending with the same 15
isolated adapters. This is expected to be close to N7-BASE functionally; a difference would
be diagnostically important. It is not a capacity-reduced baseline.

## 7. Primary attribution contrasts

Run the fidelity gate first. Only if N7-BASE passes may the following be interpreted as
source-paper attribution claims.

Report paired seed differences separately for both task orders:

1. `N7-BASE - N7-RAND-MATCH` on AP and FM: does semantically correct routing matter when
   router compute and posterior concentration are matched?
2. `N7-BASE - N7-SUM` on AP and FM: how much of the native result is attributable to routing
   rather than the compact adapter alone?
3. `N7-OID - N7-BASE` on AP and FM: remaining value of privileged task identity / router
   error headroom.
4. `N7-BASE - N7-TERM` on AP and FM: does incremental capacity allocation itself matter
   once terminal adapter capacity and router are unchanged?

Do **not** reduce AP and FM to one scalar or choose the order with the cleaner contrast.
Both orders and all three seeds are reported.

With only three reproduction seeds, confidence intervals are descriptive and must not be
sold as high-powered hypothesis tests. Report paired bootstrap intervals if the repository
metric machinery supports them, but preserve all per-seed values.

## 8. Repository metrics and accounting

In addition to source AP/FM, emit the repository's retention, plasticity and forgetting
views wherever they can be mapped unambiguously from the full task-by-time accuracy matrix.
Store that matrix so every metric can be recomputed.

For every arm report at minimum:

- `param_total`, `param_active`, `param_peak`;
- adapter bytes and router/GMM bytes;
- any retained SVD factors charged according to whether they are shared pretrained state or
  method-specific additional state;
- training FLOPs;
- inference/model FLOPs;
- GMM/K-means/covariance fitting compute;
- covariance update/inversion compute;
- `decision_flops` for likelihoods, posterior normalization and routing;
- adapter-blending compute;
- wall time and peak accelerator memory as engineering diagnostics.

The source describes GMM routing as training-free/low overhead; this protocol must still
charge its stored means/covariance and decision compute.

## 9. Implementation provenance gate

Before the first run, commit:

1. exact dataset versions/configurations and label verbalizations/prompts;
2. tokenizer/model revision;
3. preprocessing and train/validation/test split mapping;
4. software/environment lock;
5. code SHA and config hash;
6. deterministic definition of the N7-RAND-MATCH permutation;
7. an accounting note for SVD preprocessing and router covariance inversion.

If no authoritative source implementation is available, say **reimplementation from paper**.
Do not imply bit-for-bit reproduction.

## 10. Compute gate

This preregistration does **not** authorize the T5-Large run. Native T5-Large over 15 tasks,
two orders, three seeds and five inference/control arms is nontrivial compute, and the
repository's per-method compute budget is still open.

Before execution, record a cost estimate and the available hardware. The controls are mostly
inference transformations of the same trained adapter snapshots; implementation should reuse
N7-BASE trained snapshots where scientifically identical rather than retrain five copies.
Do not spend training compute merely to make arms look procedurally symmetric.

The reproduction training cost is therefore three seeds × two orders of the source training
sequence, not five independent retrainings per control.

## 11. Interpretation rules

- If N7-BASE fails the fidelity gate, report **NOT REPRODUCED**; controls may remain useful as
  mechanism diagnostics but cannot adjudicate the source paper's reported gain.
- If N7-BASE reproduces and beats N7-RAND-MATCH/N7-SUM while N7-OID adds little, that supports
  the source claim that the frozen-embedding GMM contains useful task-routing information.
- If N7-RAND-MATCH is close to N7-BASE, the apparent router gain is not attributable to the
  semantic task mapping under this control.
- If N7-OID materially improves on N7-BASE, task separability/router error remains a relevant
  limitation despite near-zero parameter forgetting.
- If N7-TERM differs materially from N7-BASE, investigate allocation-path mechanics before
  attributing the difference to routing.
- None of these outcomes re-opens the modular-consolidation architecture paper by itself.

## 12. Non-claims

- This is a methods-paper re-analysis, not EXP-100.
- It is not confirmatory evidence for CCS or consolidation.
- It does not test merge/deny/evict under a binding capacity ceiling.
- It does not select the shared Layer-A substrate.
- It does not permit post-result changes to the fidelity tolerance or source hyperparameters.


---

## Amendment 1 — implementation ambiguities resolved (2026-09-03, pre-implementation)

A final primary-source audit resolved **both** previously open questions. Details and verbatim
quotations are in `experiments/M7-LATENT-LORA-ROUTER-CONTRACT.md`.

| # | Question | Status |
| --- | --- | --- |
| A2 | Which pretrained matrix is decomposed for the SVD latent subspace? | **`SOURCE-SPECIFIED`** — each target module's own pretrained weight `W`, rank-`r` truncated SVD computed **once**, factors frozen, adapter `R_t ∈ R^{r×r}`, `ΔW_t = U_r Σ_r R_t V_r^T` (§3.2, Eq. 5) |
| A5 | Which split fits each task's GMM? | **`SOURCE-SPECIFIED`** — the task's **training** embeddings `{φ(x) : x ∈ D_t}`, fit **after** that task's adapter is trained (§3.3, Eqs. 12–13) |

Remaining items stay `OURS` and must be frozen before compute: A1 checkpoint revision pin,
A4 warmup / weight decay / gradient clipping, A6 reproduction tolerance. A3 (tokenizer and
prompt formatting) is inherited from Razdaibiedina et al. 2023 and must be pinned to that
source rather than re-invented.

No sensitivity check is owed for A2 or A5 — the source specifies both, so there is no
alternative plausible interpretation to test. **M7 is still not executed and not authorised**;
it needs an external GPU and remains behind M6 in priority.
