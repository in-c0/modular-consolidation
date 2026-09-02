# Literature and novelty audit — Modular Consolidation

**Date:** 2026-09-02
**Status:** pre-registration input. This document exists to *destroy* weak novelty claims before an experiment is designed around them.

## 0. Verification policy

Every entry below is marked with how it was checked:

- `[A]` abstract or full text retrieved and read during this audit;
- `[S]` surfaced by literature search with title + venue/identifier, summary read from search snippet only;
- `[K]` known standard reference from the field, not re-verified in this session.

Entries marked `[S]` must be upgraded to `[A]` before any claim in a submitted paper depends on them.
No claim of the form "no prior work does X" appears in this document without the corresponding negative
search being recorded in §7.

---

## 1. The prior assumption we were asked not to make

The track brief explicitly forbade starting from "dynamic experts are novel."
The audit confirms that instruction was correct. **Every individual operation in the proposed
hierarchy already exists in the literature, most of them for years.**

| Operation | Already exists | Representative prior work |
| --- | --- | --- |
| Fixed adapter | yes, standard | LoRA / adapter tuning `[K]` |
| Fixed bank of experts | yes, standard | sparse MoE `[K]` |
| Learned routing over fixed experts | yes, standard | routing networks, MoE gating `[K]` |
| Dynamic expert spawning | yes, since 2016–2018 | Progressive Nets, DEN, RCL `[K]` |
| Task-free spawning | yes | CN-DPM, Task-Free CL (Aljundi et al. 2019) `[K]`, Self-Evolved Dynamic Expansion (ICCV 2023) `[S]` |
| Spawn + merge | yes, current | MADE-IT (2026) `[A]`, MINGLE (NeurIPS 2025) `[S]`, continual model merging `[S]` |
| Spawn + prune | yes | MoCL-P (2024) `[A]`, Learn-Prune-Share `[S]`, CLNP `[S]` |
| Growth signals from saturation | yes | NORACL (2026) `[A]` |
| Autonomous module discovery | yes | Zero-Leakage Reconstruction Routing (2026) `[A]` |

**Conclusion: the architecture is not the contribution. Any paper from this track whose central
claim is "we spawn and merge modules" is dead on arrival.**

---

## 2. What the recent (2025–2026) work actually establishes

### 2.1 Expansion works, but mostly when capacity is unconstrained

The clearest statement recovered in this audit is that fixed-capacity approaches are superior
*when network capacity is held fixed*, whereas expansion-based methods win *when there is no limit
on network capacity* `[S]`. This is not a subtle caveat. It is the whole ballgame, and it is
usually reported as a background remark rather than as the object of study.

### 2.2 Capacity is reported, but almost never *controlled*

- NORACL `[A]` reports "on par with or better than the largest static baseline while using 10–22%
  fewer parameters." This is a post-hoc efficiency observation against an oracle-provisioned static
  net, not a matched-capacity control.
- MADE-IT `[A]` reports ACC and BWT only. Capacity enters through a fixed rank ratio hyperparameter
  (ρ = 0.1), not as a measured outcome.
- Zero-Leakage Reconstruction Routing `[A]` explicitly grows O(N) in tasks, never merges, never
  retires, and names routing collision at large N as unresolved.
- Several MoE-CL papers do control parameter count, typically by shrinking per-expert hidden
  dimension as expert count rises `[S]`. This is the right instinct but it is applied to the
  *number of experts* hyperparameter, not to the *dynamic allocation policy* itself.

### 2.3 Merging is treated as terminal, not as a repeated lifecycle event

Continual model merging (MADE-IT, MINGLE, ODE-perspective work) `[A][S]` treats merging as the
mechanism by which sequentially arriving models are folded into one artefact. The literature
measures the *end state* (ACC, BWT after the final merge). What it does not systematically measure:

- the immediate loss caused by an individual merge event;
- whether and how fast that loss is recovered by subsequent learning;
- whether merging *the right pair* matters more than *merging at all*.

Post-merge repair exists as a separate literature (post-merge SFT, MergeTune, continued
fine-tuning) `[S]` but is not integrated with dynamic allocation.

### 2.4 Retirement and reuse are the thinnest area

Pruning is well covered (MoCL-P `[A]`, LPS `[S]`, CLNP `[S]`, module-wise expert pruning `[S]`).
Pruning is *deletion*. What is nearly absent is **retirement with reinstatement**: taking a module
out of the active routing set and out of the active-parameter budget, retaining it in cold storage,
and later bringing it back when the stream returns to that region. Reuse is frequently *claimed*
(compositional CL, LMC, MoCL) but is measured indirectly through downstream accuracy rather than
against a ground-truth reusability structure.

### 2.5 Modularity's benefit is conditional and poorly characterised

"Dimensionality Controls When Modularity Helps in Continual Learning" (2026) `[A]` finds modularity
helps in high-dimensional settings and barely helps in low-dimensional ones, and names the
*mechanism* linking the two as open. This is a direct warning: a benchmark can be constructed in
which modularity trivially wins or trivially does not, and neither result would be informative.

---

## 3. The confounds this field has not jointly resolved

A modular continual learner that beats a baseline may be winning for any of six reasons. The audit
found ablations addressing these *individually*, inside papers advocating a specific method, but no
study that separates all six under one matched-budget protocol.

| # | Factor | Typical treatment in prior work |
| --- | --- | --- |
| F1 | **Routing** — conditional computation | ablated by disabling the router, usually without re-matching capacity |
| F2 | **Capacity** — total stored parameters | reported post-hoc; rarely an enforced ceiling |
| F3 | **Compute** — train + inference + *decision* FLOPs | rarely reported; router/detector forward passes usually uncounted |
| F4 | **Task identity** — explicit IDs or boundaries | many "task-free" methods still consume block-sequential structure |
| F5 | **Allocation** — dynamic spawning vs a fixed bank | compared against differently-sized static nets |
| F6 | **Consolidation** — merging/retirement itself | evaluated by end-state accuracy, not by event-level loss/recovery |

**The specific control that is nearly always missing is the terminal-capacity-matched fixed bank:**
run the dynamic method, observe that it converges to N modules, then train a *fixed* bank of exactly
N modules with the same router, the same compute and the same stream, and compare. Without it,
"dynamic allocation helps" is indistinguishable from "N modules is the right size."

---

## 4. What is genuinely unresolved

Stated as a question rather than as a method:

> **Does consolidation buy anything that capacity cannot?**

Expanded:

> In a task-free continual stream, when total stored parameters, active parameters per input,
> replay bytes, cold storage bytes, and total algorithmic compute (including routing/decision
> compute) are all matched, does a policy that *spawns, merges, retires and reinstates* modules
> achieve a better retention–plasticity frontier than a fixed bank of the same terminal size with
> the same learned routing?
>
> And when merging does help, is the benefit attributable to **which** modules were merged
> (the decision), to **the fact that** modules were merged (the regularisation/interference effect),
> or to the **merge operator's** approximation quality (the mechanism)?

The second half is, as far as this audit found, unaddressed. Merge quality is reported as a single
number. It is at least three separable quantities.

---

## 4a. Update after EXP-001 (2026-09-02)

The question in §4 was posed for the unbounded-capacity regime. EXP-001 established that in
that regime the retention-versus-capacity curve is monotone non-decreasing under competent
routing, so **consolidation cannot improve retention at matched capacity there at all** —
the question as posed was close to analytically settled, in the negative.

The live question is therefore the **binding-ceiling** one: under a hard ceiling below the
number of distinct skills, does pooling (merge) beat destroying (evict) or refusing (deny),
at identical capacity? EXP-002 finds a large significant gap between merge and evict, and a
null between merge and deny. No prior work found in this audit runs `deny` as a baseline or
separates merging from eviction under a fixed ceiling.

## 5. Novelty boundary — what this track may and may not claim

**May not claim as novel:**
dynamic expert allocation; task-free expert discovery; expert merging; adapter banks; routing over
adapters; pruning modules; growth triggered by saturation signals; the observation that expansion
methods bloat.

**May claim as a contribution, if the experiments support it:**

1. **N1 — A budget-matched factorial decomposition** that separates F1–F6, including the
   terminal-capacity-matched and peak-capacity-matched fixed-bank controls, and counting routing
   decision compute explicitly.
2. **N2 — Event-level consolidation dynamics.** Merge loss, merge-loss decomposition into
   *decision / interference / mechanism* components, and recovery-after-merge as measured
   quantities rather than as an end-state accuracy difference.
3. **N3 — A stream with known ground-truth module structure**, so that over-allocation and
   under-allocation are directly measurable against `K*` rather than inferred from accuracy.
4. **N4 — Retirement with reinstatement** as a distinct operation from pruning, with cold-storage
   bytes charged to the budget so that "retirement" cannot be a free lunch.
5. **N5 — Randomised consolidation controls** (merge-count-matched random-pair merging;
   spawn-count-matched random-timing spawning), isolating the *criterion* from the *rate*.

Contributions N1, N2 and N5 are methodological. That is deliberate: this track's defensible output
is a decomposition and a control protocol, not an architecture.

---

## 6. Consequences for experiment design

1. Capacity growth is a **metric**, not a footnote. A method that avoids forgetting by allocating
   unboundedly many parameters must score badly by construction.
2. Cold storage counts. Retiring a module to disk is a storage cost, not a deletion.
3. Router/detector forward passes count as decision-time compute and are reported separately
   *and* inside total algorithmic compute.
4. The benchmark must have a ground-truth `K*`, recurrence, near-duplicate segments (where merging
   is correct) and genuinely distinct segments (where merging is harmful). Otherwise merging cannot
   be scored as right or wrong.
5. Per §2.5, benchmark dimensionality must be reported and varied, because it is known to control
   whether modularity helps at all.

---

## 7. Negative searches performed

Recorded so that the absence claims in §3–§4 are auditable. Search date 2026-09-02.

| Query intent | Outcome |
| --- | --- |
| dynamic expert spawning + parameter growth + forgetting (2026) | many methods; growth reported, not controlled |
| expert merging + bounded capacity (2025–2026) | saturation–redundancy dilemma named; end-state metrics only |
| task-free expert discovery + adapter bank + routing | discovery methods found; none merge *and* retire |
| dynamically expandable / progressive / parameter isolation surveys | "uncontrolled growth of network parameters" named as the standing downside |
| expert retirement / reuse / "performance per parameter" | pruning found; retirement-with-reinstatement not found |
| capacity vs routing ablation, compute-matched, fair comparison | parameter-matched ablations exist per-paper; no cross-factor factorial found |
| post-merge repair / recovery | exists as separate literature; not integrated with allocation |
| dynamic expansion vs static net of same final size | not found as a standard control; closest is oracle-provisioned static comparison |
| factorial isolation of routing/capacity/compute/task-ID/spawn/merge | not found |

Absence of evidence in a single-day search is not proof of absence. Every `[S]`/absence claim must
be re-run and upgraded before submission, and §7 must be re-executed at freeze time.

---

## 8. Reference list

Verified during this audit (`[A]`):

- *Towards Adaptive Continual Model Merging via Manifold-Aware Expert Evolution* (MADE-IT), arXiv:2604.22464.
- *NORACL: Neurogenesis for Oracle-free Resource-Adaptive Continual Learning*, arXiv:2604.27031.
- *Modular Continual Learning via Zero-Leakage Reconstruction Routing and Autonomous Task Discovery*, arXiv:2604.14375.
- *Learn it or Leave it: Module Composition and Pruning for Continual Learning* (MoCL-P), arXiv:2406.18708.
- *Dimensionality Controls When Modularity Helps in Continual Learning*, arXiv:2606.17889.

### Upgraded to `[A]` on 2026-09-02 (abstract/landing page retrieved and read)

- **MINGLE: Mixture of Null-Space Gated Low-Rank Experts for Test-Time Continual Model
  Merging.** Qiu, Xu, He, Meng, Xu, Wu, Li. **NeurIPS 2025**, arXiv:2505.11883.
  Null-space constrained gating over low-rank experts. Reports accuracy gains (7–9%);
  **no parameter-count or capacity-growth reporting in the abstract.**
- **On Understanding of the Dynamics of Model Capacity in Continual Learning.**
  Chakraborty & Raghavan, arXiv:2508.08052 (Aug 2025). Defines **CLEMC**, an effective
  model capacity characterising the stability–plasticity balance point, and argues capacity
  in CL is non-stationary. *This is the closest prior art to our capacity framing and must
  be distinguished explicitly:* CLEMC is a property of a network's representational state;
  our `ppap`/frontier metrics are budget-accounting quantities over a lifetime. They are
  complementary, not competing, and the paper must say so.
- **When Model Merging Breaks Routing: Training-Free Calibration for MoE (HARC).**
  Huang, Shi, Quan, Wang, Zhang, Wang (2026), arXiv:2606.03391. Identifies **routing
  breakdown**: merging perturbs parameters enough that softmax/top-k routing dispatches
  tokens to the wrong experts. *Direct prior art for our `mechanism_loss` term* — it
  establishes that a real, non-trivial part of merge damage is routing damage rather than
  weight damage. Our decomposition must cite it and should consider splitting mechanism
  loss further into weight and routing components.
- **Scaling Continual Learning to 300+ Tasks with Bi-Level Routing MoE (CaRE).**
  Lou, Fu, Yu. **ICML 2026**, arXiv:2602.03473. Scales to 100–300+ tasks with two-stage
  routing. **Does not report parameter growth against task count** — supports the audit's
  §3 F2 gap at the largest task counts currently published.
- **FLAME: Adaptive Mixture-of-Experts for Continual Multimodal Multi-Task Learning.**
  Han, Chaudhari, Ranade, Chellappa, Saria (2026), arXiv:2605.09355. Keeps a **fixed-capacity**
  expert pool and adapts by compressing expert knowledge into low-rank memory while expanding
  only routers. The closest published occupant of the "compress instead of expand" cell.
- **Latent-LoRA: Compact Latent-Space Adapters with Gradient-Free Routing.**
  Azghan, Gudur, Pedrielli, Turaga, Ghasemzadeh (2026), arXiv:2607.23837. One adapter per
  task; routing by a **GMM fitted on frozen embeddings with no gradient training**. Claims
  near-zero forgetting — which, per this track's EXP-001 structural finding, is what strict
  parameter isolation gives you for free, so the interesting question is its capacity curve.
- **Unifying Detection and Adaptation in Task-Free Continual Learning (FiUni).**
  Han, Zhang, Zhu, Guo (2026), arXiv:2608.27070. Fisher-subspace matching for batch-level
  task detection; adaptively **reuses, expands, or creates** a subspace. One of the few
  methods with all three allocation outcomes in one policy.

Still `[S]` — do not cite in a paper before upgrading:

- *Dynamic Mixture of Experts Against Severe Distribution Shifts*, arXiv:2511.18987 (Kim, Nov 2025).
- *CP-MoE: Consistency-Preserving Mixture-of-Experts for Continual Learning*, arXiv:2605.20247.
- *Model Merging in LLMs, MLLMs, and Beyond* (ACM Computing Surveys, 2026) — survey, use for coverage checking.
- *LargeMonitor: Monitoring Online Task-Free Continual Learning via Large Pretrained Models*, arXiv:2606.09430.

Standard references (`[K]`): Progressive Neural Networks; Dynamically Expandable Networks;
Reinforced Continual Learning; Task-Free Continual Learning (Aljundi et al.); CN-DPM;
Local Module Composition (NeurIPS 2021); LoRA; sparse MoE gating.
