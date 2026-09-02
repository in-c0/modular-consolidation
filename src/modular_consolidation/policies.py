"""Arms and controls, expressed as one configuration object so that every arm differs
from its neighbour in exactly one factor (see docs/ARMS.md).

There is no class hierarchy here on purpose. A hierarchy invites arms that differ in
several places at once, which is the confound this track exists to remove.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

from .budget import BudgetCeiling, BudgetLedger
from .modules import Module, ModuleBank, merge_exact, merge_operator
from .toy import Stream, onehot


@dataclass
class ArmConfig:
    name: str
    routing: str = "learned"          # none | random | learned | oracle
    cap: int | None = None            # max live modules; None = unbounded
    consolidation: str = "none"       # none | merge | random_merge | full
    merge_operator: str = "operator"  # operator | exact
    novelty_z: float = -3.0
    merge_agreement: float = 0.90
    retire_idle_chunks: int = 12
    reinstate_z: float = -3.0
    consolidation_period: int = 4     # in chunks; deliberately NOT segment boundaries
    task_free: bool = True
    forced_spawn_chunks: list[int] | None = None
    forced_merge_chunks: list[int] | None = None
    extra_passes: int = 1
    seed: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArmResult:
    arm: str
    config: dict
    R: np.ndarray = field(repr=False)
    assignments: list[int] = field(repr=False)
    truth: list[int] = field(repr=False)
    route_probs: np.ndarray = field(repr=False)
    events: list[dict]
    merges: list[dict]
    ledger: dict
    k_final: int
    k_peak: int
    traffic_share: dict[int, float]
    spawn_chunks: list[int]
    merge_chunks: list[int]
    flags: list[str]

    def as_dict(self) -> dict:
        return {
            "arm": self.arm,
            "config": self.config,
            "retention_matrix": self.R.tolist(),
            "events": self.events,
            "merges": self.merges,
            "ledger": self.ledger,
            "k_final": self.k_final,
            "k_peak": self.k_peak,
            "traffic_share": self.traffic_share,
            "spawn_chunks": self.spawn_chunks,
            "merge_chunks": self.merge_chunks,
            "flags": self.flags,
        }


def _predict_with(live: dict[int, Module], phi: np.ndarray, router,
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Return (predictions, chosen module ids per sample). Never charges the ledger."""
    ids = sorted(live)
    if not ids:
        return np.zeros(phi.shape[0], dtype=int), np.full(phi.shape[0], -1)
    chosen = router(live, phi)
    preds = np.zeros(phi.shape[0], dtype=int)
    for mid in np.unique(chosen):
        sel = chosen == mid
        preds[sel] = np.argmax(live[int(mid)].predict(phi[sel]), axis=1)
    return preds, chosen


class ArmRunner:
    def __init__(self, stream: Stream, cfg: ArmConfig,
                 ceiling: BudgetCeiling | None = None):
        self.stream = stream
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.ledger = BudgetLedger(ceiling=ceiling or BudgetCeiling())
        self.bank = ModuleBank(dim=stream.cfg.d_feat, n_out=stream.cfg.n_class,
                               ledger=self.ledger)
        self.oracle_map: dict[int, int] = {}
        self.last_used: dict[int, int] = {}
        self.traffic: dict[int, int] = {}
        self.merges: list[dict] = []
        self.module_skills: dict[int, dict[int, int]] = {}
        self.spawn_chunks: list[int] = []
        self.merge_chunks: list[int] = []
        self.assignments: list[int] = []
        self.truth: list[int] = []
        self.route_prob_rows: list[np.ndarray] = []
        self.flags: list[str] = []
        self.chunk_t = 0
        # fixed random router hyperplanes (used only by routing == "random")
        k = cfg.cap or 1
        self.H = self.rng.normal(size=(k, stream.cfg.d_feat))

        if cfg.routing in ("random", "oracle"):
            if cfg.cap is None:
                raise ValueError(f"{cfg.routing} routing requires an explicit cap")
            for _ in range(cfg.cap):
                self.bank.spawn(0, reason=f"preallocated_{cfg.routing}")
        if cfg.routing == "none":
            self.bank.spawn(0, reason="single_adapter")
        if cfg.routing == "oracle" and cfg.task_free:
            self.flags.append("taskid_leak")

    # -- routers -----------------------------------------------------------
    def _router(self, charge: bool):
        cfg = self.cfg

        def route(live: dict[int, Module], phi: np.ndarray) -> np.ndarray:
            ids = sorted(live)
            if len(ids) == 1:
                return np.full(phi.shape[0], ids[0])
            if cfg.routing == "random":
                if charge:
                    self.ledger.spend_decision(len(ids), self.stream.cfg.d_feat)
                idx = np.argmax(phi @ self.H[: len(ids)].T, axis=1)
                return np.array([ids[i] for i in idx])
            if charge:
                self.ledger.spend_decision(len(ids), self.stream.cfg.d_feat)
            scores = np.stack([live[i].log_density(phi) for i in ids], axis=1)
            idx = np.argmax(scores, axis=1)
            return np.array([ids[i] for i in idx])

        return route

    def _oracle_route(self, skill: int) -> int:
        if skill not in self.oracle_map:
            self.oracle_map[skill] = sorted(self.bank.live)[skill % len(self.bank.live)]
        return self.oracle_map[skill]

    # -- allocation --------------------------------------------------------
    def _select_or_allocate(self, phi: np.ndarray, skill: int) -> Module:
        cfg = self.cfg
        if cfg.routing == "none":
            return self.bank.live[sorted(self.bank.live)[0]]
        if cfg.routing == "oracle":
            return self.bank.live[self._oracle_route(skill)]
        if cfg.routing == "random":
            self.ledger.spend_decision(len(self.bank.live), self.stream.cfg.d_feat)
            idx = int(np.bincount(np.argmax(phi @ self.H[: len(self.bank.live)].T, axis=1),
                                  minlength=len(self.bank.live)).argmax())
            return self.bank.live[sorted(self.bank.live)[idx]]

        ids, scores = self.bank.score_live(phi)
        forced = cfg.forced_spawn_chunks
        at_cap = cfg.cap is not None and len(self.bank.live) >= cfg.cap

        if forced is not None:
            want_new = self.chunk_t in forced
        else:
            if not ids:
                want_new = True
            else:
                mean_scores = scores.mean(axis=0)
                best = int(np.argmax(mean_scores))
                best_mod = self.bank.live[ids[best]]
                want_new = best_mod.novelty_z(float(mean_scores[best])) < cfg.novelty_z

        if want_new and not at_cap:
            if cfg.consolidation == "full" and self.bank.cold:
                cold_ids, cold_scores = self.bank.score_cold(phi)
                cm = cold_scores.mean(axis=0)
                bi = int(np.argmax(cm))
                cold_mod = self.bank.cold[cold_ids[bi]]
                if cold_mod.novelty_z(float(cm[bi])) >= cfg.reinstate_z:
                    return self.bank.reinstate(self.chunk_t, cold_ids[bi])
            self.spawn_chunks.append(self.chunk_t)
            return self.bank.spawn(self.chunk_t, reason="novelty" if forced is None else "forced")

        if not ids:
            self.spawn_chunks.append(self.chunk_t)
            return self.bank.spawn(self.chunk_t, reason="bootstrap")
        mean_scores = scores.mean(axis=0)
        return self.bank.live[ids[int(np.argmax(mean_scores))]]

    # -- consolidation -----------------------------------------------------
    def _functional_agreement(self, a: Module, b: Module, n: int = 128) -> float:
        """Do these two modules compute the same function on each other's input regions?"""
        phis = []
        for m in (a, b):
            if m.n < 2:
                return 0.0
            mu = m.s1 / m.n
            var = np.maximum(m.s2 / m.n - mu ** 2, 1e-4)
            phis.append(self.rng.normal(mu, np.sqrt(var), size=(n // 2, m.dim)))
        phi = np.concatenate(phis, axis=0)
        pa = np.argmax(a.predict(phi), axis=1)
        pb = np.argmax(b.predict(phi), axis=1)
        self.ledger.spend_consolidation(self.stream.cfg.d_feat)
        return float(np.mean(pa == pb))

    def _probe_accuracy(self, live: dict[int, Module], seen: int) -> float:
        router = self._router(charge=False)
        accs = []
        for i in range(seen + 1):
            seg = self.stream.segments[i]
            phi = self.stream.features(seg.Xe)
            if self.cfg.routing == "oracle":
                mid = self._oracle_route(seg.skill)
                if mid not in live:
                    continue
                preds = np.argmax(live[mid].predict(phi), axis=1)
            else:
                preds, _ = _predict_with(live, phi, router)
            accs.append(float(np.mean(preds == seg.ye)))
        return float(np.mean(accs)) if accs else 0.0

    def _maybe_consolidate(self, seen: int) -> None:
        cfg = self.cfg
        if cfg.consolidation == "none":
            return
        if len(self.bank.live) < 2:
            return

        pair = None
        if cfg.consolidation == "random_merge":
            if cfg.forced_merge_chunks is None or self.chunk_t not in cfg.forced_merge_chunks:
                return
            ids = self.bank.live_ids
            pair = tuple(self.rng.choice(ids, size=2, replace=False))
        else:
            best = (-1.0, None)
            ids = self.bank.live_ids
            for ai in range(len(ids)):
                for bi in range(ai + 1, len(ids)):
                    agree = self._functional_agreement(self.bank.live[ids[ai]],
                                                       self.bank.live[ids[bi]])
                    if agree > best[0]:
                        best = (agree, (ids[ai], ids[bi]))
            if best[1] is None or best[0] < cfg.merge_agreement:
                return
            pair = best[1]

        i, j = int(pair[0]), int(pair[1])
        record = self._measure_merge(i, j, seen)
        merged = self.bank.merge(self.chunk_t, i, j, operator=cfg.merge_operator,
                                 reason=cfg.consolidation)
        combined = dict(self.module_skills.get(i, {}))
        for k, v in self.module_skills.get(j, {}).items():
            combined[k] = combined.get(k, 0) + v
        self.module_skills[merged.mid] = combined
        self.traffic[merged.mid] = self.traffic.get(i, 0) + self.traffic.get(j, 0)
        self.last_used[merged.mid] = self.chunk_t
        self.merge_chunks.append(self.chunk_t)
        record["acc_after"] = self._probe_accuracy(self.bank.live, seen)
        self.merges.append(record)

    def _measure_merge(self, i: int, j: int, seen: int) -> dict:
        """Event-level merge accounting, including the exact-merge counterfactual.

        Counterfactual probes are measurement, not part of the method, so they are not
        charged to the ledger.
        """
        live = self.bank.live
        acc_no_merge = self._probe_accuracy(live, seen)
        a, b = live[i], live[j]

        def with_merge(fn) -> float:
            tmp = {k: v for k, v in live.items() if k not in (i, j)}
            m = fn(a, b, -1, self.chunk_t)
            tmp[-1] = m
            return self._probe_accuracy(tmp, seen)

        acc_exact = with_merge(merge_exact)
        acc_operator = with_merge(merge_operator)
        return {
            "chunk": self.chunk_t,
            "pair": [i, j],
            "same_skill": self._same_dominant_skill(i, j),
            "acc_no_merge": acc_no_merge,
            "acc_exact_merge": acc_exact,
            "acc_operator_merge": acc_operator,
            "decision_loss": acc_no_merge - acc_exact,
            "mechanism_loss": acc_exact - acc_operator,
            "total_merge_loss": acc_no_merge - acc_operator,
        }

    def _same_dominant_skill(self, i: int, j: int) -> bool | None:
        """Ground-truth check: were these two modules serving the same latent skill?

        Used only for scoring merge decisions after the fact. It is never visible to
        any policy; a run in which a policy reads this is invalid.
        """
        a = self.module_skills.get(i)
        b = self.module_skills.get(j)
        if not a or not b:
            return None
        return max(a, key=a.get) == max(b, key=b.get)

    def _maybe_retire(self) -> None:
        if self.cfg.consolidation != "full":
            return
        for mid in list(self.bank.live):
            if len(self.bank.live) <= 1:
                break
            last = self.last_used.get(mid, self.bank.live[mid].born_at)
            if self.chunk_t - last > self.cfg.retire_idle_chunks:
                self.bank.retire(self.chunk_t, mid, reason="idle")

    # -- main loop ---------------------------------------------------------
    def run(self) -> ArmResult:
        stream = self.stream
        S = stream.n_segments
        R = np.full((S, S), np.nan)
        router = self._router(charge=True)

        for t, seg in enumerate(stream.segments):
            phi_all = stream.features(seg.X)
            Y = onehot(seg.y, stream.cfg.n_class)
            for start in range(0, phi_all.shape[0], stream.cfg.chunk):
                phi = phi_all[start:start + stream.cfg.chunk]
                y = Y[start:start + stream.cfg.chunk]
                if phi.shape[0] == 0:
                    continue
                mod = self._select_or_allocate(phi, seg.skill)
                for _ in range(self.cfg.extra_passes):
                    mod.observe(phi, y)
                    self.ledger.spend_train(stream.cfg.d_feat, stream.cfg.n_class,
                                            writes=mod.deployed_params)
                self.ledger.spend_solve(stream.cfg.d_feat)
                mod.record_self_score(float(mod.log_density(phi).mean()))
                self.last_used[mod.mid] = self.chunk_t
                self.traffic[mod.mid] = self.traffic.get(mod.mid, 0) + phi.shape[0]
                ms = self.module_skills.setdefault(mod.mid, {})
                ms[seg.skill] = ms.get(seg.skill, 0) + phi.shape[0]
                self.assignments.extend([mod.mid] * phi.shape[0])
                self.truth.extend([seg.skill] * phi.shape[0])

                self.chunk_t += 1
                if self.chunk_t % self.cfg.consolidation_period == 0:
                    self._maybe_consolidate(seen=t)
                    self._maybe_retire()

            for i in range(t + 1):
                s_i = stream.segments[i]
                phi_e = stream.features(s_i.Xe)
                if self.cfg.routing == "oracle":
                    mid = self._oracle_route(s_i.skill)
                    live = self.bank.live
                    if mid in live:
                        preds = np.argmax(live[mid].predict(phi_e), axis=1)
                    else:
                        preds, _ = _predict_with(live, phi_e, router)
                else:
                    preds, chosen = _predict_with(self.bank.live, phi_e, router)
                    if i == t:
                        self.route_prob_rows.append(
                            np.bincount(chosen[chosen >= 0], minlength=self.bank._next_mid + 1)
                            .astype(float)
                        )
                self.ledger.spend_predict(
                    self.stream.cfg.d_feat * self.stream.cfg.n_class,
                    n=phi_e.shape[0])
                R[t, i] = float(np.mean(preds == s_i.ye))

        total_traffic = max(sum(self.traffic.values()), 1)
        traffic_share = {m: self.traffic.get(m, 0) / total_traffic for m in self.bank.live}

        width = max((r.size for r in self.route_prob_rows), default=1)
        probs = np.zeros((len(self.route_prob_rows), width))
        for r_i, row in enumerate(self.route_prob_rows):
            probs[r_i, : row.size] = row

        if self.cfg.routing in ("learned", "random") and self.ledger.decision_flops == 0:
            self.flags.append("uncounted_decision")
        if self.cfg.routing == "oracle":
            self.flags.append("oracle_upper_bound")

        return ArmResult(
            arm=self.cfg.name,
            config=self.cfg.as_dict(),
            R=R,
            assignments=self.assignments,
            truth=self.truth,
            route_probs=probs,
            events=list(self.bank.events),
            merges=self.merges,
            ledger=self.ledger.manifest(),
            k_final=len(self.bank.live),
            k_peak=self.ledger.live_modules_peak,
            traffic_share=traffic_share,
            spawn_chunks=self.spawn_chunks,
            merge_chunks=self.merge_chunks,
            flags=self.flags,
        )


def run_arm(stream: Stream, cfg: ArmConfig,
            ceiling: BudgetCeiling | None = None) -> ArmResult:
    return ArmRunner(stream, cfg, ceiling).run()


# ------------------------------------------------------------------ presets

def primary_arms(seed: int = 0, bank_k: int = 6) -> list[ArmConfig]:
    """A0..A6. Each differs from the previous in exactly one factor."""
    # A1 uses random routing over a bank of ONE module. Behaviourally that is a single
    # fixed adapter, but it holds the routing factor already set, so the A1 -> A2 edge
    # changes capacity alone. Writing A1 as routing="none" would change routing and
    # capacity at the same time and confound the first comparison in the ladder.
    return [
        ArmConfig("A1_single_adapter", routing="random", cap=1, seed=seed),
        ArmConfig("A2_fixed_bank_random_routing", routing="random", cap=bank_k, seed=seed),
        ArmConfig("A3_fixed_bank_learned_routing", routing="learned", cap=bank_k, seed=seed),
        ArmConfig("A4_dynamic_spawn", routing="learned", cap=None, seed=seed),
        ArmConfig("A5_spawn_merge", routing="learned", cap=None,
                  consolidation="merge", seed=seed),
        ArmConfig("A6_spawn_merge_retire", routing="learned", cap=None,
                  consolidation="full", seed=seed),
    ]


def derive_controls(results: dict[str, ArmResult], seed: int = 0) -> list[ArmConfig]:
    """Controls whose configuration is a deterministic function of a realised run.

    This mirrors the State Promotion convention of deriving the count-matched random
    control from the target arm's own manifest, so the control cannot be chosen after
    seeing which comparison is favourable.
    """
    out: list[ArmConfig] = []
    a4 = results.get("A4_dynamic_spawn")
    a5 = results.get("A5_spawn_merge")
    a6 = results.get("A6_spawn_merge_retire")

    if a4 is not None:
        out.append(ArmConfig("C-TERM(A4)", routing="learned",
                             cap=a4.k_final, seed=seed))
        out.append(ArmConfig("C-RSPAWN(A4)", routing="learned", cap=None,
                             forced_spawn_chunks=_random_times(len(a4.spawn_chunks),
                                                               a4, seed),
                             seed=seed))
        out.append(ArmConfig("C-OID(A4)", routing="oracle", cap=max(a4.k_final, 1),
                             task_free=False, seed=seed))
    if a5 is not None and a5.merge_chunks:
        out.append(ArmConfig("C-RMERGE(A5)", routing="learned", cap=None,
                             consolidation="random_merge",
                             forced_merge_chunks=list(a5.merge_chunks), seed=seed))
    if a6 is not None:
        out.append(ArmConfig("C-TERM(A6)", routing="learned",
                             cap=a6.k_final, seed=seed))
        out.append(ArmConfig("C-PEAK(A6)", routing="learned",
                             cap=a6.k_peak, seed=seed))
    return out


def _random_times(n: int, ref: ArmResult, seed: int) -> list[int]:
    rng = np.random.default_rng(seed + 991)
    horizon = max(max(ref.spawn_chunks, default=1), 1)
    total = max(horizon, n)
    return sorted(int(x) for x in rng.choice(np.arange(total + 1), size=min(n, total + 1),
                                             replace=False))
