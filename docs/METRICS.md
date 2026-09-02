# Metric definitions — Modular Consolidation

All metrics are defined here once so that arms, ablations and the validator share one vocabulary.
Notation: the stream is a sequence of segments `s = 1..S`. `acc(i, t)` is accuracy on segment `i`'s
held-out evaluation set measured at stream position `t`. `T` is the end of the lifetime.

## 1. Behaviour

| Metric | Definition | Direction |
| --- | --- | --- |
| `avg_acc` | mean over `i` of `acc(i, T)` | higher better |
| `retention` | mean over `i < S` of `acc(i, T)` | higher better |
| `forgetting` | mean over `i` of `max_t acc(i,t) − acc(i,T)` | lower better |
| `plasticity` | mean over `i` of `acc(i, t_i^end)`, immediately after segment `i` is learned | higher better |
| `retention_auc` | mean over `i` of the area under `acc(i, ·)` from `t_i^end` to `T` | higher better |
| `bwt` | mean over `i<S` of `acc(i,T) − acc(i, t_i^end)` | higher better |

`plasticity` and `retention` are the two axes of the **primary frontier**. Neither alone is a result.

## 2. Capacity — first-class, not a footnote

| Metric | Definition |
| --- | --- |
| `param_total` | all parameters that must exist for the system to run, including every live module |
| `param_active` | expected parameters touched per input at inference, under the arm's realised routing distribution |
| `param_peak` | maximum `param_total` at any point in the lifetime |
| `param_added` | `param_total(T) − param_total(0)` |
| `state_bytes` | consolidation state (sufficient statistics, optimizer state, provenance) |
| `cold_bytes` | bytes held for retired modules. **Retirement is not free.** |
| `storage_total` | `param_total·b + state_bytes + cold_bytes + replay_bytes` |

A method that reduces forgetting by growing `param_total` without bound is scored as failing,
by construction, through §4.

## 3. Compute

Reported separately and then summed, following the State Promotion convention that decision-time
routing is not hidden:

| Metric | Definition |
| --- | --- |
| `train_flops` | forward+backward for parameter updates |
| `infer_flops` | forward passes producing predictions |
| `decision_flops` | router / novelty-detector / merge-affinity forward passes and comparisons |
| `consolidation_flops` | merge, compression and reinstatement operations |
| `total_algorithmic_flops` | sum of the above |
| `param_writes` | count of parameter elements written over the lifetime |

`decision_flops` grows with live module count for any O(N) router. This is a real cost of allocation
policies and must not be excluded when claiming compute parity.

## 4. Efficiency frontier — the primary reporting object

| Metric | Definition |
| --- | --- |
| `ppap` | *performance per added parameter*: `(avg_acc − avg_acc_frozen) / max(param_added, 1)` |
| `ppac` | performance per active parameter: `avg_acc / param_active` |
| `retention_per_byte` | `retention / storage_total` |

Primary result is reported as a **frontier plot**, retention against `param_total` and against
`total_algorithmic_flops`, with every arm on the same axes. A method is only interesting if it is on
the Pareto front. Scalar `ppap` is a summary of that plot, not a substitute for it.

## 5. Routing and specialisation

Let `r(x)` be the module selected (or the routing distribution) for input `x`, and `g(x)` its
ground-truth generating skill.

| Metric | Definition |
| --- | --- |
| `routing_entropy` | mean over inputs of `H(r(·|x))`; 0 = hard routing, high = diffuse |
| `specialisation_nmi` | normalised mutual information `I(r; g) / H(g)` |
| `module_purity` | mean over modules of the max share of a single skill among its routed inputs |
| `router_accuracy` | agreement between selected module and the module that would minimise loss |

`specialisation_nmi` is the honest version of "our experts specialised." Routing entropy alone can be
low for a router that is confidently wrong.

## 6. Allocation quality — requires ground truth `K*`

Available because the benchmark (CAMS-v0) is generated from a known skill library of size `K*`.

| Metric | Definition |
| --- | --- |
| `k_final` | live module count at `T` |
| `k_peak` | maximum live module count |
| `over_allocation` | `max(0, k_final − K*) / K*` |
| `under_allocation` | `max(0, K* − k_final) / K*` |
| `allocation_error` | `|k_final − K*| / K*` |
| `spawn_precision` | fraction of spawn events that occurred at a true novel-skill onset |
| `spawn_recall` | fraction of true novel-skill onsets that triggered a spawn |

## 7. Consolidation dynamics — the event-level contribution

For a merge event `m` at time `t_m` over module set `M`, evaluated on a probe set fixed before the
event:

| Metric | Definition |
| --- | --- |
| `merge_loss(m)` | `probe_acc(t_m − ε) − probe_acc(t_m + ε)` |
| `merge_recovery(m, k)` | fraction of `merge_loss(m)` recovered after `k` subsequent stream steps |
| `merge_recovery_time(m)` | steps until 90% of `merge_loss(m)` is recovered, or censored |
| `permanent_merge_loss(m)` | unrecovered loss at `T` |

### 7.1 Merge-loss decomposition

`merge_loss` is decomposed into three separable components. This decomposition is only possible
because the toy learner admits an **exact** merge (see §7.2).

| Component | Definition | Interpretation |
| --- | --- | --- |
| `mechanism_loss` | `acc(exact merge of M) − acc(operator merge of M)` | how lossy the merge *operator* is |
| `decision_loss` | `acc(no merge) − acc(exact merge of M)` | cost of merging things that should not be merged |
| `interference_delta` | change in accuracy on *other* modules' segments caused by the merge | collateral effect |

A merge policy can be excellent at deciding and terrible at merging, or the reverse. Reporting a
single "merge hurt by x%" number conflates them.

### 7.2 Exact merge reference

For a ridge/least-squares module holding sufficient statistics `A = ΦᵀΦ`, `b = Φᵀy`, `n`, the merge
`A_i + A_j`, `b_i + b_j`, `n_i + n_j` yields exactly the model that would be obtained by fitting the
union of both modules' data. This is the `exact merge` reference. The `operator merge`
(e.g. sample-weighted parameter averaging) is the practical mechanism under test. The gap between
them is `mechanism_loss` and is measurable without any counterfactual retraining.

## 8. Reuse and retirement

| Metric | Definition |
| --- | --- |
| `reuse_rate` | fraction of segments served by a pre-existing module with no spawn |
| `reinstatement_rate` | fraction of retired modules later returned to the live set |
| `reinstatement_precision` | fraction of reinstatements where the reinstated module was the best available |
| `cold_residency` | mean fraction of lifetime a module spends retired |
| `zombie_rate` | fraction of live modules receiving < 1% of routed traffic over the last window |

`zombie_rate` catches the failure where a policy nominally supports retirement but never uses it.

## 9. Validity flags (validator-enforced)

A run is `INVALID` if any of the following hold. Invalid runs are archived, never silently dropped.

- `ceiling`: all adaptive arms exceed 95% accuracy before segment 3.
- `floor`: any arm fails to exceed chance on the segment it is currently training on.
- `taskid_leak`: an arm declared task-free consumed segment identity, boundary, or ordering metadata.
- `budget_breach`: `param_total`, `storage_total` or `total_algorithmic_flops` exceeded the declared ceiling.
- `uncounted_decision`: `decision_flops == 0` for an arm that performs routing.
- `unmatched_control`: a capacity-matched control's `param_total` differs from its target by > 2%.
- `seed_reuse`: a confirmatory seed appears in the development seed list.
