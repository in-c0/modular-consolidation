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
