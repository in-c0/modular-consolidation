# Native-fidelity ledger (Layer B)

**Date: 2026-09-03. Preparation only — no native reproduction has been run, and none is
authorised by this document.**

Layer B asks a different question from Layer A and neither substitutes for the other:

- **Layer A** — standardized mechanism panel: what happens to each *mechanism* under one
  shared substrate, so factors are comparable across methods.
- **Layer B** — native-fidelity re-analysis: does a *published gain* survive faithful
  reconstruction on the paper's own substrate plus the missing attribution controls.

**Only Layer B may adjudicate a paper's published gain.** A transplanted mechanism that
scores differently on a shared substrate says nothing about whether the original paper's
claim holds.

## Evidence discipline applied here

Every entry separates: (a) what the **paper** specifies; (b) what the **official code**
does; (c) what **we** would have to introduce. Settings we introduce are marked `OURS` and
must be frozen before execution. No algorithmic detail is inferred from reported output
behaviour, and where paper and code conflict the conflict is recorded rather than resolved
by whichever reproduces the headline number.

Primary sources were retrieved and read as PDFs on 2026-09-03; code repositories were
inspected through the GitHub API on the same date.

---

## Status summary

| ID | Method | Status |
| --- | --- | --- |
| M1 | Fixed bank + learned router (constructed reference) | `NOT_APPLICABLE` (Layer A only; no published gain to adjudicate) |
| M2 | `BEXP-LoRA` boundary-expansion reference | `NOT_APPLICABLE` (Layer A only; explicitly not a DEN reproduction) |
| M3 | MoCL-P | `BLOCKED_SOURCE_AMBIGUITY` |
| M4 | Zero-Leakage Reconstruction Routing | `BLOCKED_CODE_PAPER_CONFLICT` |
| M5 | MADE-IT | `BLOCKED_SOURCE_AMBIGUITY` (no code released) |
| M6 | NORACL | **`READY_FOR_PREREG`** |
| M7 | Latent-LoRA | `PREREGISTERED / UNRUN` (paper-derived reimplementation; no code released) |
| M8 | FLAME | `NATIVE_ONLY` + `BLOCKED_DATA_ACCESS` |

---

## M1 — Fixed bank + learned router (constructed reference)

| Field | Value |
| --- | --- |
| Primary paper | none — constructed by us to occupy the "routing, no allocation, no consolidation" cell |
| Official code | n/a |
| Native benchmark | n/a |
| Status | `NOT_APPLICABLE` for Layer B |

M1 exists because no published method cleanly occupies that cell. It has no published gain,
so it cannot be adjudicated. It belongs to Layer A only.

## M2 — `BEXP-LoRA` boundary-expansion reference

| Field | Value |
| --- | --- |
| Primary paper | none — a boundary-expansion reference, **explicitly not a Progressive Networks or DEN reproduction** |
| Status | `NOT_APPLICABLE` for Layer B |

Naming discipline: describing it as "DEN" would be a false attribution. It is a reference
arm, not a reconstruction of any paper.

---

## M3 — MoCL-P

**Status: `BLOCKED_SOURCE_AMBIGUITY`.** Final source-recovery attempt completed 2026-09-03;
see `experiments/M3-MOCLP-FIDELITY-NOTE.md` for the full audit.

| Field | Value | Provenance |
| --- | --- | --- |
| Primary paper | Wang, Adel, Lange, Strötgen, Schütze — *Learn it or Leave it*, RepL4NLP 2024, arXiv:2406.18708 | paper |
| Official code | `github.com/boschresearch/MoCL-Pruning` (paper footnote 1) — **HTTP 404 on 2026-09-03**; no forks, no tags, no snapshot recovered | GitHub API |
| Predecessor code | `boschresearch/MoCL-NAACL-2024`, public but **archived**; implements MoCL, not MoCL-P | GitHub API |
| Native benchmark | MTL15 (15 classification tasks), plus AfriSenti and WikiAnn | paper |
| Native backbone | **T5-large** for MTL15 (AfroXLMR for AfriSenti, BERT for WikiAnn) | paper §5.2 |
| Adaptation mechanism | **prefix-tuning**, prefix length 10 for MTL15 | paper §4.1, Table 5 |
| Routing / composition | per-instance matching `α_i = cos(x_n, v_i)` against learnable task feature vectors; composition `P'_m = Σ_{k=0..m} α_k P_k` | paper §4.2–4.3 |
| Capacity mechanism | one module per task, then adaptive pruning of the new module | paper §4.3 |
| Consolidation | prune the newly trained module if `α_m < α_ths` | paper §4.3, §6.4 |
| Training task IDs | yes — task-incremental source protocol | paper |
| Inference task IDs | no — "per-instance task module matching and composition" | paper §4.4 |
| Training config (MTL15) | AdamW; 40 epochs; early-stop patience 5; batch 8; lr 5e-2; max seq len 512; prefix length 10; A100 | paper Table 5, A.2.1 |
| Data | 1000 samples/class train, 500/class held-out validation | paper §5.1 |
| Orders | three (O1/O2/O3), exact sequences in paper Table 6 | paper |
| Seeds | three random seeds | paper §5.2 |
| `α_ths` | **0.25** for MTL15 (0.025 for AfriSenti and WikiAnn) — benchmark-selected operating point, not a universal law | paper §6.4 |
| Published targets (MTL15, T5-large) | MoCL-P **AVG 82.5** (O1 83.0, O2 82.7, O3 81.8), **15.6M ±1.1** params; MoCL 82.5 / 21.1M; Per-task FT 80.5 / 21.1M; ProgPrompt 77.9; O-LoRA 69.6; EPI 65.4; Seq FT-P 64.7 / 1.4M | paper Table 2 |

### The blocker, restated precisely

`α_i(x) = cos(x_n, v_i)` is defined **per input**, and §4.4 confirms matching is per-instance
at inference. The pruning rule compares a **single scalar** `α_m` to `α_ths`. The paper never
states the reduction from the per-example matching weights to that scalar. Verbatim, §6.4:
"we compare the matching weight of the newly initialized task module `α_m` with the
pre-specified threshold `α_ths`, if `α_m < α_ths`, then we discard the newly learned module."

Candidate reductions (task mean, median, max, a batch statistic, train vs validation split)
give different keep/discard decisions near the threshold, so the choice is part of the causal
mechanism, not a coding detail.

**New corroborating evidence, insufficient to close the contract.** The archived predecessor
implementation maintains a per-task running mean of batch-mean matching weights:

```python
# boschresearch/MoCL-NAACL-2024, model/mtl_prefix_encoder.py
if self.attn_weights[task_id] is None:
    self.attn_weights[task_id] = w_avg
else:
    self.attn_weights[task_id] = ((self.attn_weights[task_id] * self.steps_val[task_id])
                                  + w_avg) / (self.steps_val[task_id] + 1)
```

This is consistent with the third-party "mean coefficient" description, and `steps_val`
suggests the statistic is accumulated over validation steps. **It is corroboration, not
authority**: it comes from MoCL, which has no pruning step, so it cannot establish what
MoCL-P compares to `α_ths`. Per the standing rule, the gap is **not** filled by adopting the
most natural implementation.

**Disposition: `NATIVE FIDELITY: BLOCKED — AUTHORITATIVE ALGORITHM UNDER-SPECIFIED`.**
Everything except the scalar reduction is now frozen above, so recovery of one detail — via
the released code or an author clarification — would move M3 straight to `READY_FOR_PREREG`.

---

## M4 — Zero-Leakage Reconstruction Routing

**Status: `BLOCKED_CODE_PAPER_CONFLICT`.** Full side-by-side in
`experiments/M4-PAPER-VS-CODE-CONTRACT.md`.

| Field | Value | Provenance |
| --- | --- | --- |
| Primary paper | N. Kermiche (Western Digital), arXiv:2604.14375 | paper |
| Official code | `github.com/norikermiche-123/Modular_Continual_Learning` — 7 standalone scripts, two commits (2026-03-29), 0 stars | GitHub API |
| Native benchmark | Split-MNIST (**2 tasks**: digits 0–4 vs 5–9) and a **synthetic "Crowded Manifold" dataset** simulating 4096-D LLaMA-3 embeddings | paper §7.1, §7.3 |
| Published targets | Task A retention after Task B: Ours 99.42%, Replay 95.10%, EWC 84.00%, LwF 79.80%, Naive 19.40%; routing-ablation and bottleneck-sweep tables | paper Tables 1–3 |
| Training task IDs | none claimed (autonomous discovery) | paper |
| Inference task IDs | none — Contrastive Soft Routing with OOD rejection | paper §5.6 |

**Two independent problems, both disqualifying for Layer B.**

1. **Paper and code encode materially different algorithms** on the novelty threshold,
   calibration, commitment gate and inference routing. Details in the contract document.
2. **Scope.** Even a perfect reconstruction would adjudicate a 2-task Split-MNIST retention
   number and a synthetic-manifold ablation. There is no long-sequence continual-learning
   benchmark gain of the kind Layer B exists to test.

Two provenance-labelled candidates are defined (`M4-PAPER-CONTRACT`, `M4-CODE-CONTRACT`), and
**neither can be called a reproduction of the published result**: the code does not implement
the paper's stated mechanism, and the paper's mechanism is not the one that produced the
released scripts' behaviour. Native published-gain adjudication for M4 is **blocked**.

---

## M5 — MADE-IT

**Status: `BLOCKED_SOURCE_AMBIGUITY`** — no code released. Do **not** translate to LoRA.

| Field | Value | Provenance |
| --- | --- | --- |
| Primary paper | Qiu, Wu, Tan (HK PolyU) — *Towards Adaptive Continual Model Merging via Manifold-Aware Expert Evolution*, arXiv:2604.22464 | paper |
| Official code | **none found** — no repository link, no availability statement, no anonymised release in the paper | paper + search |
| Native benchmark | CLIP-ViT continual **model merging**, 8 / 14 / 20 task sequences | paper §4 |
| Native backbone | CLIP **ViT-B/32, ViT-B/16, ViT-L/14** | paper Table 1 |
| Adaptation units | rank-`r` SVD experts from module weight updates: `E_c^(t) = Ũ Σ̃ Ṽ^T`, `r = ⌊ρ·min(d_o, d_i)⌋` | paper §3.1.1 Eq. 5 |
| Merge operation | subspace merging on consolidation, guided by projection-based subspace affinity with a distribution-aware adaptive threshold | paper §3.1 |
| Routing | **data-free, training-free implicit routing** via feature-projection alignment, with a task-identity constraint propagated over an expert dependency graph anchored at the highest-diversity module | paper §3.2, Fig. 1 |
| Capacity mechanism | consolidate (merge) when affinity exceeds the adaptive threshold; create only when uniqueness justifies expansion | paper §3.1 |
| Task IDs | none at inference (implicit routing); merging consumes arriving task-specific *models*, not task labels | paper |
| Hyperparameters | rank ratio **ρ = 0.1**, margin coefficient **β = 1.0**, global across architectures and sequences | paper §4 |
| Seeds | 10 runs, seeds **42–51**, one task order each | paper §4 |
| Published targets (ACC%, ViT-B/32 / B/16 / L/14 at 8·14·20 tasks) | **87.1±1.6, 84.3±0.6, 81.4±1.3** / 90.4±0.6, 87.7±0.4, 83.0±0.1 / 93.1±0.7, 91.7±0.7, 89.6±0.7; best baseline MINGLE 85.8 / 81.6 / 77.1 (B/32) | paper Table 1 |
| Metrics | ACC and BWT, mean±std over ten task orders | paper Table 1 |

### Native attribution controls, classified

| Control | Class | Note |
| --- | --- | --- |
| Merge-count-matched **random-pair** subspace merging | native-compatible causal ablation | isolates the affinity criterion from merging itself; keeps SVD experts and implicit routing intact |
| **Fixed-`ρ` no-consolidation** expert bank at MADE-IT's realised final expert count | native-compatible causal ablation | the terminal-capacity control the paper does not run |
| **Oracle task-ID routing** replacing implicit routing | native-compatible causal ablation | prices the training-free router; upper bound, must carry the `oracle_upper_bound` flag |
| Parameter / storage / **decision-compute** accounting across all arms | accounting-only comparison | paper reports ACC and BWT only; adds no mechanism |
| Replacing SVD experts with **LoRA adapters** | **invalid** — destroys native fidelity | the merge operates on Grassmann-manifold subspaces of `ΔW`; LoRA substitution changes the object being merged |
| Substituting a learned gate for implicit routing | **invalid** | training-free routing is the paper's contribution, not an implementation detail |

**Blocker:** without released code, the adaptive-threshold rule, the dependency-graph
construction and the anchor-selection rule would all have to be reconstructed from prose.
That is more than one unresolved detail, so M5 cannot yet claim source-paper fidelity.

---

## M6 — NORACL

**Status: `READY_FOR_PREREG`.** Official implementation located on 2026-09-03 at
`github.com/karthik-charan/NORACL` (first author, "Official implementation", `CITATION.cff`,
last push 2026-08-12). This was not found in the earlier audit and materially changes M6's
disposition.

| Field | Value | Provenance |
| --- | --- | --- |
| Primary paper | Raghunathan, Metzner, Kriener, Payvand (Institute of Neuroinformatics) — arXiv:2604.27031 | paper |
| Official code | `karthik-charan/NORACL`, **69 YAML configs** including ablations | GitHub API |
| Native benchmark | Permuted MNIST, Rotated MNIST, Binary Split MNIST; Binary Split CIFAR-10 in configs | paper Table 1, configs |
| Native backbone | 1-layer and 2-layer **MLPs**, 12.7k–54.9k parameters | paper Table 1 |
| Adaptation mechanism | dense MLP training with **online EWC**; capacity added by **neuron insertion**, not by adapters | paper §3, config `ewc: true` |
| Growth trigger | `grow_l = (φ_l > γ·φ_l^(0)) ∧ (Percentile(F_l^(curr), p) > τ_l)` — ED saturation **and** Fisher saturation | paper Eq. 6 |
| Neuron insertion | fan-in from a random orthogonal basis via **QR decomposition**, scaled by `s_init = 0.2`; fan-out **initialised to zero** (function-preserving); Fisher diagonal and EWC anchors **zero-padded**; optimizer state re-initialised | paper §3.2 |
| Cool-down | `C` epochs after each growth event | paper §3.1 |
| Routing | **none** — a single growing network, no experts, no router | paper |
| Task IDs | not used for architecture decisions; hyperparameters explicitly independent of `T`, task similarity and task id | paper §3.3 |
| Frozen config (`bsmnist_2l_noracl.yaml`) | `gamma: 0.9`, `f_sat_percentile: 25`, `alpha: 0.9`, `init: qr_init`, `qr_init_scale: 0.2`, `hidden_dim: 12`, `k_fixed: 2`, `lr: 0.005`, `lr_1: 0.1`, `lr_boost_multiplier: 3.0`, `n_epochs: 30`, `n1_epochs: 10`, `annealing_epochs: 3`, `batch_size: 256`, `importance: 5000`, `orth_thresh: 0.05`, `ewc: true`, `n_tasks: 5`, `seed: 0` | official code |
| Published targets | Permuted MNIST 1L: NORACL **79.9±0.5** at 47.6k±1.6 params vs static-large 76.0±0.8 at 50.8k; 2L: 79.4±0.7 at 49.2k±3.2 vs 73.3±1.8 at 54.9k. Rotated MNIST 1L 72.6±2.4 / 42.2k. Binary Split MNIST 1L 72.1±1.8 / 23.8k | paper Table 1 |

### Native attribution controls

The official repo already ships most of them as `growth_trigger` and `init` switches, which
means they can be run **without us modifying the method**:

| Control | Class | Mechanism |
| --- | --- | --- |
| `growth_trigger: ed_only` | native-compatible causal ablation | isolates the ED half of the trigger |
| `growth_trigger: fsat_only` | native-compatible causal ablation | isolates the Fisher half |
| `growth_trigger: loss_plateau` | native-compatible causal ablation | the heuristic the paper argues against |
| `growth_trigger: random` / `fixed_pertask` | native-compatible causal ablation | **growth-count-matched** controls — isolates the trigger from the growth rate |
| `init: he` / `xavier` / `nullspace` vs `qr_init` | native-compatible causal ablation | isolates function-preserving initialisation from expansion itself |
| `static16/32/64/128/…/1024` configs | native-compatible causal ablation | **capacity-matched static baselines**, including at NORACL's realised final width |
| Parameter/compute/storage accounting on every arm | accounting-only | paper reports params and accuracy; we add compute and storage |
| Replacing neuron insertion with **spawning a LoRA expert** | **invalid** — destroys native fidelity | NORACL grows *neurons inside layers*; an adapter bank is a different mechanism and must not be called NORACL |

**The one control the paper does not appear to run** is the growth-count-matched random-trigger
comparison *at the realised final width*, i.e. separating "grew at the right time" from "ended
at the right size". The `random` and `fixed_pertask` configs make it available natively.

---

## M7 — Latent-LoRA

**Status: `PREREGISTERED / UNRUN`.** Prereg exists (`769a8cd`,
`experiments/M7-NATIVE-REANALYSIS-PREREG.md`); router contract frozen in `237debe`. Audited
against the primary source on 2026-09-03 — see the readiness report below.

| Field | Value | Provenance |
| --- | --- | --- |
| Primary paper | Azghan, Gudur, Pedrielli, Turaga, Ghasemzadeh — arXiv:2607.23837 | paper |
| Official code | **none found** — no repository link, no availability or reproducibility statement, no supplementary release | paper + search |
| Native benchmark | **Long Sequence** (Razdaibiedina et al. 2023) Orders **3 and 4**; SuperNI Orders 1 and 2 | paper Table 1 |
| Native backbone | **T5-Large**; also T5-XLarge, Llama-2-7B, Llama-3-8B | paper §4 |
| Adaptation mechanism | one LoRA adapter per task, constrained to the **principal subspace of the pretrained weights via SVD** (latent-space parameterisation) | paper §3 |
| Routing | **gradient-free GMM** over mean-pooled frozen input-embedding vectors `φ(x) = MeanPool(Emb(x))`; posterior `p(t\|x)` under a uniform prior blends all adapters, `R(x) = Σ_t p(t\|x) R_t` | paper Eqs. 11, 14, 15 |
| Capacity mechanism | one adapter per task, frozen as a snapshot after training; no merging, no pruning | paper §3 |
| Consolidation | none | paper |
| Training task IDs | yes (per-task training) | paper |
| Inference task IDs | **no** — task-agnostic setting | paper §4.1 |
| Adapter config | Latent-LoRA `r=32, α=16`; all LoRA baselines `r=8, α=16`; adapters on **query and value** projections of each block | paper C.3 |
| Training | AdamW; T5 lr **3e-4**, batch **8**; constant LR schedule; **30 epochs per task** on T5 (15 on Llama) | paper C.3 |
| Orthogonal regularisation | `λ = 0.02` for **Long Sequence** (0.05 for SuperNI), selected on a held-out validation split of T5-Large | paper C.3 |
| GMM router | **K = 5** components per task, initialised by **K-means with 30 iterations**, shared covariance regularisation **ε = 0.01**, soft routing at inference; fitting takes 2–5 s per task on **CPU** | paper C.3 |
| Seeds | three, averaged | paper §4.1 |
| Published targets (T5-Large) | **Long Sequence O3 AP 79.95 / FM 0.57; O4 AP 79.87 / FM 0.73**. SuperNI O1 48.60/0.01, O2 49.75/0.01. Best baseline GainLoRA: LS O3 78.21/0.72, O4 76.26/1.14 | paper Table 1 |

### Execution-readiness report

Every value already frozen in our prereg and router contract was re-checked against the
primary source and **all are source-derived**: `r=32`, `α=16`, Q/V targeting, SVD-latent
parameterisation, AdamW, 30 epochs/task, `λ=0.02` for Long Sequence, GMM `K=5`, K-means 30
iterations, `ε=0.01`, soft posterior composition, Orders 3 and 4, and the AP/FM targets. No
opportunistic change was made.

Remaining ambiguities, all of which must be frozen as `OURS` **before** compute:

| # | Ambiguity | Status |
| --- | --- | --- |
| A1 | Exact `t5-large` checkpoint revision pin | `OURS` — pin a revision hash at freeze time; the paper cites Raffel et al. only |
| A2 | Which pretrained weight matrix the SVD principal subspace is taken from per adapter, and whether the basis is recomputed per task | **paper-specified in prose; needs a written-out equation in our prereg before coding** |
| A3 | Tokenizer/prompt formatting for Long Sequence | inherited from Razdaibiedina et al. 2023; must be pinned to that source, not re-invented |
| A4 | Warmup / weight decay / gradient clipping | not stated; `OURS`, freeze explicit defaults |
| A5 | Whether the GMM is fitted on train or held-out embeddings | not stated; `OURS`, freeze before running |
| A6 | Reproduction tolerance | `OURS` — propose ±1.0 AP absolute against 79.95 / 79.87, frozen before execution |

A2 and A5 are the two that could change results materially and must be resolved in writing
first. **M7 is not cleared for compute until A1–A6 are frozen.**

---

## M8 — FLAME

**Status: `NATIVE_ONLY` and `BLOCKED_DATA_ACCESS`.** Do not force it into a LoRA benchmark.

| Field | Value | Provenance |
| --- | --- | --- |
| Primary paper | Han, Chaudhari, Ranade, Chellappa, Saria (JHU) — arXiv:2605.09355 | paper |
| Official code | **`github.com/aaronhan223/FLAME/tree/continual-learning`** — branch verified to exist on 2026-09-03; MNIST demo also linked | paper footnote 2, GitHub API |
| Native benchmark | **healthcare multimodal**: MIMIC-IV + MIMIC-CXR (48-IHM, LOS, 25-PHENO), eICU (MOR, RAD), EMBED (BIRADS, RISK, DENSITY), ADNI — four datasets, nine tasks | paper §3.1 |
| Adaptation mechanism | **fixed-size** expert pool; each new task compressed into a **low-rank additive slice**; only lightweight task-specific routers expand | paper §1, abstract |
| Routing | modality-specific routers over a shared expert pool; cursor-based inference | paper |
| Capacity mechanism | fixed expert pool — capacity does **not** grow with tasks; router growth only | paper |
| Consolidation | low-rank compression of accumulated expert knowledge into memory subspaces | paper |
| Claimed gains | structural no-forgetting guarantee via cursor-based inference at **5–15× fewer parameters** than fine-tuning, EWC and LoRA; competitive multitask pretraining | paper §1 |

### Which claims can and cannot be mapped to the D6 lattice

| Claim | Mappable? | Reason |
| --- | --- | --- |
| Parameter efficiency (5–15× fewer than FT/EWC/LoRA) | **yes** | direct capacity accounting; our `param_total` / `storage_total` axes apply unchanged |
| Structural no-forgetting from cursor-based inference | **yes** | retention/forgetting are measurable, and a compression-off control is native-compatible |
| Compression is justified because functional energy is low-rank | **yes, as an accounting-only comparison** | the rank sweep is already in the paper; we add storage/compute columns |
| Gains attributable to **routing** vs **capacity** | **partially** | routing is modality-specific and the expert pool is fixed, so our capacity factor is constant by construction — the `C-TERM` control is degenerate here, exactly as it was for spawn-only arms in EXP-003 |
| Gains attributable to **consolidation vs eviction** | **no** | FLAME never evicts; there is no slot-freeing decision, so the EXP-002/EXP-003 contrast has no analogue |
| Cross-method frontier comparison against M1/M2/M7 | **no** | different modalities, different metrics, different data; forcing it onto Long Sequence would be a substrate transplant mislabelled as re-analysis |

**Data-access blocker.** MIMIC-IV, MIMIC-CXR and eICU require PhysioNet credentialing and a
signed data-use agreement; EMBED and ADNI require separate application and approval. These
are **owner-only actions** — they involve identity verification and legal agreements that I
cannot and should not complete. Until credentials exist, M8 native reproduction is blocked
regardless of code availability.

---

## Compute and execution cost table

Estimates for planning only. No run is authorised by this table.

| ID | Model | Trainable params | Tasks | Epochs/steps | Seeds | Control evals | Controls reuse checkpoints? | Est. GPU memory | Est. storage | Practical target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M6 | 1L/2L MLP, 12.7k–55k params | all (dense) | 5 (BSMNIST) / 10+ (P-MNIST) | 30 epochs/task, batch 256 | 3+ | ~10 configs × arms | **no** — each trigger/init variant is a separate training run | **< 1 GB**; CPU-feasible | < 100 MB | **local Mac (M1 Max)** — the only method currently runnable here |
| M7 | T5-Large (~770M) | LoRA r=32 on Q/V ≈ 9–10M/task | 15 (Long Sequence) | 30 epochs/task | 3 | 4 control arms | **partly** — oracle-router and accounting controls reuse trained adapters; capacity-matched and no-orthogonality arms need retraining | ~24–40 GB for bf16 training at batch 8, seq 512 | adapters ≈ 10M × 15 tasks × 3 seeds × 2 orders ≈ 3–4 GB + checkpoints | **external GPU (A100/H100 class)**; not the Mac |
| M3 | T5-large, prefix len 10 | ≈ 1.4M/task | 15 | 40 epochs/task, batch 8, lr 5e-2 | 3 | 3–4 | partly | ~24–40 GB | ~2–3 GB | external GPU — **blocked** on the pruning statistic |
| M5 | CLIP ViT-B/32, B/16, L/14 | merging only, no training | 8 / 14 / 20 | none (training-free merging) | 10 orders | 3–4 | **yes** — merging consumes pre-existing fine-tuned checkpoints | modest for B/32; L/14 heavier | dominated by **downloading 20 fine-tuned CLIP checkpoints per architecture** | external GPU — **blocked** on missing code |
| M4 | small AE/MLP | tiny | 2 | short | any | n/a | n/a | < 1 GB | negligible | trivially runnable, **but scientifically blocked** |
| M8 | multimodal MoE | fixed pool | 9 across 4 datasets | not extracted | not extracted | limited (see mapping table) | unknown | large (imaging + time series) | large — MIMIC-CXR and EMBED are imaging corpora | external GPU + **credentialed data access (owner-only)** |

**Consequence.** M6 is the only method that is both scientifically cleared and cheap enough
to run locally. M7 is the only other one close to cleared, and it needs an external GPU plus
resolution of A1–A6. Everything else is blocked on sources, code or data access.

## Compute authorization boundary

M6 is now `READY_FOR_PREREG`, not "ready to run". The next steps that do **not** cross the
boundary are: writing the M6 native prereg with its control set and reproduction tolerance
frozen; verifying the official repo runs at all with a score-free mechanical smoke test; and
recording environment provenance.

**Launching the M6 reproduction, or any T5-Large run, is an execution decision and is left to
the owner.**
