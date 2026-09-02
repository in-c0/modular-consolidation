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
