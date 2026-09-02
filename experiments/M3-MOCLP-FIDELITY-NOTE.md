# M3 / MoCL-P — source-fidelity note

**Status: PRE-RUN SOURCE AUDIT — M3 remains blocked.**

Date: 2026-09-02

This note resolves what can and cannot currently be recovered authoritatively for Wang et
al., *Learn it or Leave it: Module Composition and Pruning for Continual Learning*
(arXiv:2406.18708 / RepL4NLP 2024). No M3 result has been run.

## What the paper specifies clearly

The paper defines a learned task feature vector `v_i` in the same representation dimension
as the input embedding and computes an **instance-level** matching score

`alpha_i(x) = cos(x, v_i)`.

During training on a new task, the new module is composed with older frozen modules using the
matching weights. The optimization objective includes a term that increases similarity
between examples of the current task and the new task feature vector. After training, the
new module's matching weight `alpha_m` is compared with a pruning threshold; if it is below
the threshold the new module is discarded.

The paper also makes two fidelity facts important for this repository:

- its evaluated implementation uses prefix tuning; adapters/LoRA are described as compatible
  PEFT alternatives in principle, not evaluated MoCL-P variants;
- its experimental setting is task-incremental and uses task information in the source
  protocol, so D6 task-agnostic Layer A necessarily requires an explicit transplant and an
  oracle `C-OID` path rather than being a native reproduction.

For MTL15 the source reports `alpha_ths = 0.25`. The threshold is not presented as a universal
constant: the paper studies the performance/parameter tradeoff over thresholds and selects a
benchmark setting. For source-fidelity work we therefore preserve `0.25`; we do **not**
retune it on our outcomes. In Layer A it must be described as a **source-derived,
benchmark-selected operating point**, not as a preregistered universal pruning law.

## The unresolved executable detail

The paper defines `alpha_m(x)` per input example but, in the pruning description, refers to
comparing `alpha_m` with the scalar threshold after task training. The paper text available
to us does not specify the reduction that maps the set of per-example matching weights to
that one pruning statistic.

Plausible reductions include a task mean, median, maximum, a batch statistic, or a statistic
computed on a particular split. They can produce different prune/keep decisions near the
threshold. Choosing one is therefore part of the causal mechanism, not an innocuous coding
detail.

**Rule:** do not infer the reduction from notation and do not choose it by downstream
performance or desired pruning rate.

## Code provenance audit

The paper points to:

`https://github.com/boschresearch/MoCL-Pruning`

As checked on 2026-09-02, that repository is not publicly retrievable through GitHub and no
public repository/fork named `MoCL-Pruning` was found in a repository search.

The public predecessor repository
`boschresearch/MoCL-NAACL-2024` is archived and implements the earlier MoCL work. Its README
explicitly says the follow-up MoCL-P code would be released later; a code search there does
not expose the missing pruning statistic.

A third-party technical summary describes the pruning value as an average matching
coefficient over task instances. That is a useful hypothesis for recovery, but it is **not
authoritative provenance** and cannot close the implementation contract by itself.

## Consequence for the two D6 evidence layers

### Layer A — standardized mechanism panel

M3 remains **specified but not executable**. Admission requires both:

1. an authoritative or independently justified pruning-statistic contract committed before
   any M3 score; and
2. the task-agnostic/common-substrate adaptation required by `plasticity-routing`.

If the original code or author clarification becomes available and establishes the reduction,
freeze it verbatim. If it cannot be recovered, a future standardized mechanism may define a
new explicit reduction (for example a task mean) **only under a new descriptive transplant
name** and with a pre-run justification; it must not be presented as faithful MoCL-P.

### Layer B — native-fidelity re-analysis

Do not claim source-paper reproduction while the executable prune rule is unknown. Two
scientifically acceptable paths remain:

- recover the original MoCL-P implementation / author clarification; or
- predeclare a paper-derived reimplementation with the unresolved reduction explicitly
  labelled as an assumption, then classify fidelity as insufficient for claims that a
  published MoCL-P gain survives or fails our controls.

The second path can still be useful for mechanism diagnostics, but not for source-paper
attribution.

## What is *not* a blocker

The missing statistic does not block the methods paper as a whole. D6 deliberately permits a
smaller valid standardized panel and native-fidelity work on methods whose executable
contracts are recoverable. M1/M2/M7 remain the clean Layer-A starting core once the shared
substrate is exported by `plasticity-routing`.

## Source anchors

- paper: https://arxiv.org/abs/2406.18708
- published proceedings: https://aclanthology.org/2024.repl4nlp-1.12/
- paper-linked MoCL-P repository: https://github.com/boschresearch/MoCL-Pruning
- public predecessor implementation: https://github.com/boschresearch/MoCL-NAACL-2024

This note records provenance and ambiguity only; it contains no experimental evidence.
