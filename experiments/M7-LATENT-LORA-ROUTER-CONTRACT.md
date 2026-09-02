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
