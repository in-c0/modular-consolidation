# Terminology

Terms are fixed here so that this track, its sibling repositories and the eventual
umbrella programme use one vocabulary. Where a term already has a settled meaning in the
literature, that meaning is kept.

| Term | Meaning in this track |
| --- | --- |
| **module** | A parameterised unit that can be independently allocated, routed to, merged, retired and reinstated. An adapter, an expert, a low-rank head. Not a layer. |
| **bank** | The set of modules that exist. Partitioned into *live* and *cold*. |
| **live** | In the active routing set; costs `param_total` and can be selected. |
| **cold** | Retired; costs `cold_bytes` only; cannot be selected until reinstated. |
| **spawn** | Creating a new module. Increases capacity. |
| **merge** | Combining two or more modules into one. Reduces live module count. |
| **compress** | Reducing a module's parameter count without changing the module count. Not exercised in EXP-000; reserved. |
| **retire** | Moving a live module to cold storage. **Not deletion** — the bytes are still charged. |
| **prune** | Deleting a module permanently. Distinct from retirement; the literature usually means this when it says pruning. |
| **reinstate** | Returning a cold module to the live set. |
| **reuse** | Serving a stream segment with a pre-existing module rather than spawning. |
| **allocation** | The policy governing spawn decisions. |
| **consolidation** | The policy governing merge, compress, retire and reinstate. |
| **routing** | Selecting which live module handles an input. Must be label-free at inference. |
| **decision compute** | FLOPs spent choosing (routing, novelty detection, merge affinity). Counted separately and included in the total. |
| **terminal capacity** | `param_total` at the end of a lifetime. |
| **peak capacity** | Maximum `param_total` during a lifetime. |
| **capacity-matched control** | A fixed-capacity arm whose size is derived from a target arm's realised capacity. |
| **merge loss** | Accuracy dropped by one merge event, measured on a probe fixed before the event. |
| **decision loss** | The part of merge loss attributable to merging the wrong modules, measured with an ideal merge operator. |
| **mechanism loss** | The part attributable to the merge operator's approximation. |
| **recovery** | Fraction of merge loss regained by subsequent learning. |
| **`K*`** | Ground-truth number of distinct latent skills in a synthetic stream. |
| **task-free** | No segment identity, boundary or ordering metadata reaches the policy. Enforced, not asserted. |

## Terms deliberately avoided

- *"Catastrophic forgetting avoided"* without a capacity number attached. Any method can
  avoid forgetting by growing.
- *"Experts specialised"* on the basis of routing entropy or NMI alone. See EXP-000
  Finding 1.
- *"Parameter-efficient"* without stating which of `param_total`, `param_active`,
  `param_peak` or `storage_total` is meant.
