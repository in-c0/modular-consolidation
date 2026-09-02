# Benchmark policy

## The rule

Benchmark structure — the skill library, the recurrence and near-duplicate pattern, the
segment ordering distribution — is fixed before confirmatory seeds are drawn and is never
adjusted afterwards. In particular it is never adjusted after seeing which arm wins.

## The one permitted adjustment: difficulty calibration

A benchmark at ceiling or at floor cannot answer anything, so difficulty may be calibrated.
To keep that from becoming a back door, calibration uses a criterion that mentions only
arms which are **not candidate methods**:

* `A1` — a single adapter, the no-modularity floor;
* `C-OID` — oracle task-ID routing over a bank of `K*`, the routing upper bound.

A configuration is admissible when the headroom between them is at least 0.15 and the
single adapter is not above 0.95. Among admissible settings, the **least difficult** one is
chosen, so that difficulty is not inflated until modularity looks good.

This is implemented in `scripts/calibrate_stream.py` and its output is committed to
`results/calibration/`.

## Admissibility criterion for the interference regime (predeclared 2026-09-02)

EXP-000 showed that a stream of *separated* input regions is solved by capacity alone: a
fixed bank of the right size matched or beat every allocation and consolidation policy. Such
a stream cannot distinguish consolidation from capacity, so it cannot answer this track's
question no matter how carefully the arms are controlled.

CAMS-v1 therefore adds an **interference** knob: latent skills are made to share input
regions, so that distinct skills demand conflicting mappings from the same inputs, and a
weak context signal is the only route to telling them apart.

**This criterion is fixed before the sweep is run, and is committed in a separate commit
that precedes the commit containing any sweep output.** It references only arms that are
not candidate consolidation methods.

A CAMS-v1 configuration is **admissible** when all four hold, averaged over development
seeds:

| # | Condition | Purpose |
| --- | --- | --- |
| K1 | `A1` single-adapter retention `<= 0.95` | no ceiling |
| K2 | `A1` single-adapter retention `>= 0.35` | no floor; the stream must be learnable at all |
| K3 | `retention(C-OID) - retention(A1) >= 0.15` | modularity has headroom |
| K4 | `retention(C-OID) - retention(A2) >= 0.10` | **capacity with a bad router is not enough** |

K4 is the new condition and the one that makes the regime useful. In EXP-000's stream a
fixed bank with random routing was already close to the oracle once it had enough modules,
which is precisely why capacity explained everything. A configuration that fails K4 is
rejected regardless of how interesting its consolidation results look.

Among admissible configurations the **least interfering** one is chosen, so that difficulty
is not escalated until consolidation starts to look good.

Arms referenced: `A1` (single adapter), `A2` (fixed bank, random routing), `C-OID` (oracle
task-ID upper bound). None of them spawns, merges, retires or reinstates. No candidate
consolidation policy influences admissibility.

### Amendment A — K4 is insufficient; add K5 (2026-09-02)

**What happened.** The K1–K4 sweep was run and *every* interference setting passed,
including `interference = 0`. Checking K4 against EXP-000's own separated-region stream
shows it would have passed there too (oracle 0.867 vs random-routed bank 0.630, gap 0.237,
comfortably above the 0.10 threshold). A criterion that admits the stream already shown to
be incapable of answering the question is not doing its job.

**Why it failed.** K4 tests whether **routing** beats capacity. That was never in doubt --
EXP-000 already showed learned routing beating random routing by 18 points. The thing that
made capacity explain everything in EXP-000 was different: **retention was monotone
increasing in the number of modules.** When more modules is always better, no allocation or
consolidation policy can beat "just use the largest bank", so the benchmark cannot
distinguish consolidation from capacity however well the arms are controlled.

**The amendment.** A fifth condition is added:

| # | Condition | Purpose |
| --- | --- | --- |
| K5 | over a sweep of fixed-bank caps `K` with learned routing, `max_K retention(K) - retention(K_max) >= 0.05` | **over-allocation must actually cost something** |

That is: the retention-versus-capacity curve must have an interior maximum. If it is
monotone, the correct policy is trivially "allocate the maximum", and there is nothing for
consolidation to do.

**Why this is not criterion-shopping.** The amendment is made on evidence about the
*criterion's discriminating power*, not about which consolidation policy wins. At the time
of this amendment **no consolidation arm (A5, A6, or any C-SHRINK/C-RMERGE control) has
been run on CAMS-v1 at all**, so K5 cannot have been reverse-engineered toward an outcome.
The git history is the evidence: this amendment is committed before any CAMS-v1
consolidation run exists.

K5 references arm `A3` (fixed bank, learned routing) at several caps. `A3` does not spawn,
merge, retire, reinstate or compress, so it remains a non-candidate arm.

K4 is **retained, not deleted**. It is a necessary condition that happens not to be
sufficient, and recording that is more useful than quietly replacing it.

**If no configuration satisfies K1–K5**, that is reported as a negative instrument result.
The thresholds are not relaxed to manufacture an admissible stream.

### Amendment B — K5 is unsatisfiable; the unbounded-capacity regime cannot answer the question (2026-09-02)

**What happened.** K5 (over-allocation must cost at least 0.05 retention) failed at every
setting tried, across three independent mechanisms, each swept with `A3` only (fixed bank,
learned routing -- not a consolidation method):

| mechanism swept | range | best K5 cost |
| --- | --- | --- |
| interference (skills sharing input regions) | 0.00 – 1.00 | 0.010 |
| data scarcity (samples per segment / feature dim) | 10.0 – 1.0 | 0.024 |
| soft routing (density-gated mixture over all live modules) | interference 0.0 – 1.0 | 0.000 |

The capacity curves are monotone non-decreasing and then flat. Example, hard routing at
interference 0: K=1 → 0.675, K=2 → 0.738, K=4 → 0.801, K=8 → 0.828, K=16 → 0.839,
K=24 → 0.839.

**The diagnosis is structural, not a generator bug.** In a parameter-isolated modular
system with competent routing, a module that is never selected cannot damage a prediction.
Spare modules therefore cost parameters, compute and storage — but not accuracy. Soft
routing was tested precisely because it is the mechanism by which spare modules *could*
dilute a prediction, and it changed nothing: density gating in 48 dimensions is so peaked
that the mixture is effectively hard. Blurring it further with temperature would only
reproduce `A2`, a deliberately bad router.

**Consequence — the primary hypothesis was malformed.** If retention is monotone in
capacity, then consolidation *cannot* improve retention at matched capacity. There is
nothing for it to fix. The original H1 ("A6 beats C-TERM(A6) on retention at equal
capacity") is close to analytically false for this architecture class, which is why EXP-000
found what it found. Consolidation's only possible benefit in the unbounded regime is
moving left along the efficiency frontier at equal retention — that is a **compression**
claim, not a **forgetting** claim, and it should be stated as one.

**Reformulation — the binding-ceiling regime.** Consolidation can only be irreducible to
capacity when capacity cannot simply be increased. Under a hard ceiling smaller than the
number of distinct skills, a policy that wants a new module must free a slot, and *how* it
frees that slot is a real decision that capacity accounting cannot make for it:

* **deny** — refuse to spawn, keep using an existing module;
* **evict** — delete a module and spawn a fresh one;
* **merge** — pool two modules into one and spawn a fresh one.

All three end each step at exactly the same live-module count, the same parameter count and
the same storage. Capacity is equal *by construction*, so any difference between them is
attributable to consolidation and to nothing else. This is also the realistic deployment
case: fixed memory, unbounded task stream.

**K6 replaces K5** (K1–K4 retained):

| # | Condition | Purpose |
| --- | --- | --- |
| K6a | `ceiling < K*` | the ceiling must actually bind |
| K6b | `retention(A3, unbounded) - retention(A3, ceiling) >= 0.05` | the ceiling must cost something, so the slot decision matters |

K6 is structural and references only `A3` and the stream's own `K*`. It is decidable
without running any consolidation policy.

**Why this is not criterion-shopping.** As with Amendment A, no consolidation arm has been
run on CAMS-v1 at the time of this amendment. The evidence that prompted it is about
whether *any* policy could differ, not about which one wins. K5 is retained in the record
as an unsatisfiable condition, and the reason it is unsatisfiable is itself the finding.

## Seed discipline

- Development and calibration seeds: 900–999.
- Confirmatory seeds: drawn from a disjoint range, with the selection rule recorded before
  the first confirmatory run.
- A run whose seed appears in both sets is invalid (`seed_reuse`).

## Ground-truth fields are for scoring only

`Segment.skill`, `Stream.k_star`, `is_novel_onset` and the `same_skill` field on merge
records exist so that allocation and merge decisions can be scored after the fact. A policy
that reads any of them is invalid. Only `C-OID`, which is declared an upper bound and
carries the `oracle_upper_bound` flag, is permitted access to task identity.

## Reporting invalid runs

Invalidated runs are archived with their flags and appear as a count with reasons in any
write-up. They are never silently dropped.
