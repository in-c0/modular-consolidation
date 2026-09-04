# M7 — Latent-LoRA router contract for the standardized panel

**Status: SOURCE-DERIVED DESIGN CONTRACT — UNRUN.**

This freezes the M7 routing mechanism before the shared adapter substrate is selected by
`in-c0/plasticity-routing`. It separates the source paper's GMM router from its compact
SVD-latent adapter geometry so the methods paper can attribute routing and adapter effects
instead of conflating them.

Source: Rahimi Azghan et al., *Latent-LoRA: Compact Latent-Space Adapters with
Gradient-Free Routing for Continual Learning* (arXiv:2607.23837v1).

## Frozen benchmark/orders

Use Long Sequence Orders 3 and 4 exactly as recorded in
`experiments/STANDARDIZED-TRANSPLANTS.md`. Inference is task-agnostic. Task identity is
available only for scoring and `C-OID`.

## Frozen router representation

For each input `x`, compute the router representation from the **frozen base model input
embedding layer**:

`phi(x) = MeanPool(Emb(x))`

No adapter output, hidden state after adapted attention, learned projection or gradient-updated
router representation may substitute for this in the source-derived M7-router arm. The
point is that the representation remains stationary as adapters are learned.

## Frozen per-task GMM

After training task `t`'s adapter, fit one task-specific GMM over that task's `phi(x)` values:

- `K = 5` Gaussian components per task;
- initialise component means with K-means;
- K-means iterations: `30`;
- use one covariance matrix shared across all tasks/components;
- covariance is the pooled within-task scatter plus `epsilon * I`;
- `epsilon = 0.01`;
- no gradient updates to router parameters.

When a new task is added, fit only that task's component means and update the sufficient
statistics required for the shared covariance. Do not retain old raw examples merely to
refit the router.

## Frozen inference rule

Use a uniform task prior. For each input, compute task likelihood under each task GMM and
normalise over known tasks:

`p(t|x) = p(phi(x)|t) / sum_j p(phi(x)|j)`

Use the resulting posterior for **soft routing**, not argmax selection, in the source-derived
arm. The effective adapter is the posterior-weighted blend of the stored task adapters.

## Separation from adapter geometry

The source Latent-LoRA adapter is not ordinary LoRA: it uses a compact trainable `r x r`
matrix inside frozen SVD factors, with source rank `r=32` and an orthogonal regulariser. The
standard LoRA baselines in the source use rank `r=8`.

Therefore Layer A must expose at least these distinct cells once the common adapter substrate
is fixed:

1. **COMMON-ADAPTER + M7-GMM** — same adapter family/rank as the other standardized arms,
   with only the source GMM router transplanted;
2. **COMMON-ADAPTER + matched random router** — router control;
3. **COMMON-ADAPTER + C-OID** — oracle routing control;
4. **native Latent-LoRA** — Layer B/source-fidelity path, preserving its compact SVD adapter
   and source training hyperparameters.

Do not call cell 1 “Latent-LoRA” if the compact adapter geometry is absent; call it
`M7-GMM-router` or equivalent.

## Accounting

Count as stored/router state:

- all GMM component weights and means;
- shared covariance / sufficient statistics retained to update it;
- any task-index metadata required for mixture lookup.

Count all likelihood evaluation, normalisation and adapter blending work in
`decision_flops`/inference compute. The router is gradient-free, not compute-free or
storage-free.

## No-tuning rule

`K=5`, 30 K-means iterations and `epsilon=0.01` are source-derived constants and are frozen
for the primary standardized M7-router run. A sensitivity study may vary them only under a
separately named, predeclared analysis and may not replace an unfavourable primary result.

## Remaining dependency

This contract closes the M7 **router** definition. It does not select the common backbone,
adapter family/rank, optimization budget or target modules; those remain gated on
`in-c0/plasticity-routing`.


---

## Source resolution of the two open implementation questions (2026-09-03)

Both were resolved from the primary source (arXiv:2607.23837) by reading §3.2, §3.3 and
Appendix C.3 in full. Neither is a reimplementation choice any more; both are
**`SOURCE-SPECIFIED`**.

### Q1 — which pretrained weight matrix is decomposed? **RESOLVED**

§3.2, verbatim: "We adopt a more compact parameterization based on **LoRA-XS** (Bałazy et
al., 2025), which constrains weight updates to the principal subspace of the pretrained
weights. Given a **target weight** `W ∈ R^{m×n}`, we compute its rank-`r` truncated SVD
`W ≈ U_r Σ_r V_r^T` **once** and keep all three factors **frozen**."

So:

- the decomposed matrix is **the target module's own pretrained weight matrix** — i.e. each
  targeted query and value projection weight, decomposed separately, not a shared or
  aggregated matrix;
- the SVD is computed **once** and the factors are **frozen**; it is **not** recomputed per
  task;
- the per-task trainable object is `R_t ∈ R^{r×r}` and the update is
  **`ΔW_t = U_r Σ_r R_t V_r^T`** (Eq. 5).

Source location: §3.2 "Compact Latent-Space Adapters", Eq. 5.

### Q2 — which samples fit each task's GMM? **RESOLVED**

§3.3, verbatim: "**After training the adapter for task `t`**, we fit a task-specific Gaussian
mixture model (GMM) over the **training embeddings** `{φ(x) : x ∈ D_t}`."

So:

- the GMM is fit on the **task's training split** `D_t`, not a validation or held-out split;
- it is fit **after** that task's adapter finishes training, not before;
- the shared covariance is the regularised pooled within-task scatter
  `C = (Σ_t S_t) / (Σ_t n_t) + εI` (Eq. 13), where `n_t` is the number of training examples
  for task `t` — confirming the training split independently.

Source location: §3.3 "Training-Free Task Router", Eqs. 12–13.

The held-out validation split mentioned in Appendix C.3 is used only to select the orthogonal
regularisation coefficient `λ`, **not** to fit the router. Conflating the two would have been
the natural error, and it is now excluded by the text.

### Consequence

The M7 native contract has **no remaining algorithmic ambiguity**. What remains is
environmental and is `OURS`: the `t5-large` checkpoint revision pin, warmup/weight-decay/
gradient-clipping defaults not stated by the paper, and the reproduction tolerance. The
sensitivity obligation previously anticipated for Q1/Q2 is **discharged** — no alternative
interpretation needs testing, because the source specifies both.
