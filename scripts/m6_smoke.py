#!/usr/bin/env python3
"""M6 — MECHANICAL VALIDATION, NOT SCIENTIFIC EVIDENCE.

Score-free smoke check for the official NORACL implementation at the pinned
revision. It verifies *plumbing*, never a hypothesis:

* the pinned checkout is present, correct and unmodified;
* dataset and model construction work;
* growth can occur at all;
* each selected growth-trigger path executes;
* each selected initialization path executes;
* growth events serialize;
* seeding is deterministic;
* accounting fields are populated.

It deliberately cannot answer anything in §B5 of the preregistration: it uses
1-2 tasks and a handful of epochs, and it **never** compares end performance
across scientific arms. Accuracy numbers it happens to observe are printed only
as liveness evidence and are explicitly not results.

Run with the native environment, e.g.:

    third_party/venv-noracl/bin/python scripts/m6_smoke.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modular_consolidation.native import (  # noqa: E402
    NORACL_PIN,
    GrowthEventRecorder,
    Instrumentation,
    derive_c_match_hidden_dim,
    derive_t_count_k_fixed,
    mlp_param_count,
)
from modular_consolidation.native.noracl import (  # noqa: E402
    GROWTH_TRIGGERS,
    INIT_STRATEGIES,
    ed_fired,
    width_delta,
)

BANNER = "MECHANICAL VALIDATION — NOT SCIENTIFIC EVIDENCE"

# The smoke subset: one trigger per code path we intend to run, one init per
# code path. Chosen for coverage of branches, not for scientific comparison.
SMOKE_TRIGGERS = ("ed_fisher", "ed_only", "fisher_only", "fixed_pertask")
SMOKE_INITS = ("qr_init", "random", "xavier")


def checkout(dest: pathlib.Path) -> dict:
    sha = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(dest), "status", "--porcelain"],
                           capture_output=True, text=True, check=True).stdout.strip()
    return {"sha": sha, "clean": not dirty, "matches_pin": sha == NORACL_PIN["sha"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(ROOT / "third_party" / "noracl"))
    ap.add_argument("--tasks", type=int, default=2, help="1-2 only; smoke is not a run")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "results" / "m6_smoke"))
    args = ap.parse_args()

    if args.tasks > 2 or args.epochs > 3:
        print("refusing: that is a scientific run, not a smoke test")
        return 2

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dest = pathlib.Path(args.repo)
    report: dict = {"banner": BANNER, "pin": NORACL_PIN, "checks": {}}

    print("=" * 78)
    print(BANNER)
    print("=" * 78)

    # 1. checkout integrity ------------------------------------------------
    ck = checkout(dest)
    report["checks"]["checkout"] = ck
    print(f"\n[1] checkout {ck['sha'][:12]} matches_pin={ck['matches_pin']} clean={ck['clean']}")
    if not (ck["matches_pin"] and ck["clean"]):
        print("    FAIL — official checkout must be the pinned revision and unmodified")
        return 3

    sys.path.insert(0, str(dest))

    # 2. imports and identifier sets --------------------------------------
    import yaml  # noqa: E402
    from noracl.core.growth import GROWTH_TRIGGERS as SRC_TRIGGERS  # noqa: E402
    from noracl.core.init import INIT_STRATEGIES as SRC_INITS  # noqa: E402
    from noracl.models import mlp as mlp_mod  # noqa: E402

    ok_ids = (tuple(SRC_TRIGGERS) == GROWTH_TRIGGERS
              and tuple(SRC_INITS) == INIT_STRATEGIES)
    report["checks"]["identifiers"] = {
        "source_triggers": list(SRC_TRIGGERS), "source_inits": list(SRC_INITS),
        "match_frozen": ok_ids}
    print(f"[2] identifiers match frozen contract: {ok_ids}")
    print(f"    triggers {tuple(SRC_TRIGGERS)}")
    print(f"    inits    {tuple(SRC_INITS)}")
    if not ok_ids:
        print("    FAIL — the pinned source disagrees with the frozen identifier sets")
        return 4

    # 3. config plumbing ---------------------------------------------------
    cfg_path = dest / "configs" / "bsmnist_2l_noracl.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    report["checks"]["config"] = cfg
    print(f"[3] config {cfg_path.name}: dataset={cfg['dataset']} n_tasks={cfg['n_tasks']} "
          f"hidden={cfg['hidden_dim']} trigger={cfg['growth_trigger']} init={cfg['init']}")

    # 4. published-target plumbing ----------------------------------------
    pt = dest / "results" / "paper" / "bsmnist_2l_noracl_s0" / "per_task.csv"
    rows = [ln.split(",") for ln in pt.read_text().strip().splitlines()[1:]]
    final = rows[-1]
    widths = [int(final[2]), int(final[3])]
    published_params = int(final[4])
    recomputed = mlp_param_count(cfg["n_in"], widths, cfg["n_out"])
    report["checks"]["param_model"] = {
        "published_params": published_params, "recomputed": recomputed,
        "agree": published_params == recomputed}
    print(f"[4] parameter model vs published: {recomputed} == {published_params} "
          f"-> {published_params == recomputed}")

    # 5. derived controls --------------------------------------------------
    k_fixed = derive_t_count_k_fixed([cfg["hidden_dim"]] * cfg["n_layers"], widths,
                                     cfg["n_tasks"])
    h_match = derive_c_match_hidden_dim(published_params, cfg["n_in"], cfg["n_layers"],
                                        cfg["n_out"])
    report["checks"]["derived_controls"] = {
        "T_COUNT_k_fixed": k_fixed, "C_MATCH_hidden_dim": h_match,
        "C_MATCH_params": mlp_param_count(cfg["n_in"], [h_match] * cfg["n_layers"],
                                          cfg["n_out"]),
        "C_MATCH_residual": mlp_param_count(cfg["n_in"], [h_match] * cfg["n_layers"],
                                            cfg["n_out"]) - published_params}
    print(f"[5] derived controls (seed 0): T-COUNT k_fixed={k_fixed}, "
          f"C-MATCH hidden_dim={h_match} "
          f"(residual {report['checks']['derived_controls']['C_MATCH_residual']:+d} params)")

    # 6. dataset plumbing --------------------------------------------------
    t0 = time.time()
    from noracl.data.mnist import load_binary_split_mnist_dataloader  # noqa: E402
    train_dl, test_dl = load_binary_split_mnist_dataloader(0, cfg["batch_size"])[:2]
    xb, yb = next(iter(train_dl))
    report["checks"]["dataset"] = {
        "loaded": True, "batch_x": list(getattr(xb, "shape", [])),
        "batch_y": list(getattr(yb, "shape", [])),
        "load_seconds": round(time.time() - t0, 2)}
    print(f"[6] dataset ok: batch {tuple(xb.shape)} / {tuple(yb.shape)} "
          f"in {time.time() - t0:.1f}s")

    # 7. model + growth machinery -----------------------------------------
    import jax  # noqa: E402
    from noracl.core.init import resolve_init_fn  # noqa: E402

    init_paths = {}
    key = jax.random.PRNGKey(0)
    params = mlp_mod.init_params(cfg["n_in"], cfg["hidden_dim"], cfg["n_out"],
                                 cfg["n_layers"], key) \
        if hasattr(mlp_mod, "init_params") else None
    for name in SMOKE_INITS:
        try:
            fn = resolve_init_fn(name, key, params, scale=cfg["qr_init_scale"])
            arr = fn((cfg["hidden_dim"], 2))
            init_paths[name] = {"ok": True, "shape": list(arr.shape)}
        except Exception as exc:  # recorded, not silenced
            init_paths[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    report["checks"]["init_paths"] = init_paths
    print("[7] initialization paths: " +
          ", ".join(f"{k}={'ok' if v['ok'] else 'FAIL'}" for k, v in init_paths.items()))

    # 8. instrumentation on the real module -------------------------------
    from noracl.training import loop as loop_mod  # noqa: E402
    rec = GrowthEventRecorder(out / "smoke_events.jsonl", arm="SMOKE", seed=0)
    instr = Instrumentation(rec)
    instr.install(loop_mod)
    patched = [a for a in ("neurogenesis_step", "trigger_act", "resolve_init_fn")
               if hasattr(loop_mod, a)]
    instr.uninstall()
    report["checks"]["instrumentation"] = {"patchable_symbols": patched}
    print(f"[8] instrumentation attaches to loop bindings: {patched}")

    # 9. serialization round trip -----------------------------------------
    from modular_consolidation.native.noracl import GrowthEvent  # noqa: E402
    probe = GrowthEvent(task=0, epoch=1, trigger_mode="ed_fisher", init_mode="qr_init",
                        phi_curr=[0.95, 0.4], phi_0=[0.8, 0.8], gamma=cfg["gamma"],
                        widths_before=[12, 12], widths_after=[14, 12],
                        params_before=mlp_param_count(784, [12, 12], 2),
                        params_after=mlp_param_count(784, [14, 12], 2), grew=True)
    rec.record(probe)
    path = rec.flush()
    back = GrowthEventRecorder.load_jsonl(path)
    report["checks"]["serialization"] = {
        "events": len(back), "ed_fired_derived": ed_fired(back[0]),
        "width_delta": width_delta(back[0]), "path": str(path)}
    print(f"[9] event serialization: {len(back)} record(s), derived ed_fired="
          f"{ed_fired(back[0])}, width_delta={width_delta(back[0])}")

    # 10. deterministic seeding -------------------------------------------
    a = jax.random.normal(jax.random.PRNGKey(7), (4,))
    b = jax.random.normal(jax.random.PRNGKey(7), (4,))
    same = bool((a == b).all())
    report["checks"]["determinism"] = {"same_key_same_draw": same}
    print(f"[10] deterministic seeding: {same}")

    # 11. every selected growth-trigger path executes ---------------------
    import jax.numpy as jnp  # noqa: E402
    from noracl.core.growth import neurogenesis_step  # noqa: E402
    from noracl.models.mlp import init_model_weights, model  # noqa: E402

    def fresh():
        return init_model_weights(cfg["n_in"], cfg["n_out"], cfg["n_layers"],
                                  [cfg["hidden_dim"]] * cfg["n_layers"],
                                  key=jax.random.PRNGKey(0))

    trigger_paths = {}
    rec2 = GrowthEventRecorder(out / "smoke_growth_events.jsonl", arm="SMOKE-GROWTH", seed=0)
    for trig in SMOKE_TRIGGERS:
        params = fresh()
        # `fisher` and `params_prev` are lists of PER-TASK structures
        fisher = [[jnp.zeros_like(p) for p in params]]
        # phi_curr well above gamma*phi_0 so an ED-gated trigger has reason to fire
        phi_0 = [0.30] * (cfg["n_layers"] + 1)
        phi_curr = [0.95] * (cfg["n_layers"] + 1)
        M = [p.shape[1] for p in params]
        tau = [0.0] * (cfg["n_layers"] + 1)
        init_fn = resolve_init_fn(cfg["init"], jax.random.PRNGKey(1), params,
                                  scale=cfg["qr_init_scale"])
        try:
            before = [p.shape for p in params]
            new_params, _, _ = neurogenesis_step(
                params, fisher, [jnp.zeros_like(p) for p in params],
                [[jnp.zeros_like(p) for p in params]],
                phi_0, phi_curr, M, cfg["gamma"], init_fn, tau,
                cfg["f_sat_percentile"], trigger=trig, k_fixed=cfg["k_fixed"])
            after = [p.shape for p in new_params]
            grew = before != after
            trigger_paths[trig] = {"ok": True, "grew": grew,
                                   "widths_before": [int(s[1]) for s in before[:-1]],
                                   "widths_after": [int(s[1]) for s in after[:-1]]}
            from modular_consolidation.native.noracl import GrowthEvent as _GE
            rec2.record(_GE(trigger_mode=trig, init_mode=cfg["init"],
                            phi_curr=phi_curr, phi_0=phi_0, gamma=cfg["gamma"],
                            widths_before=[int(s[1]) for s in before[:-1]],
                            widths_after=[int(s[1]) for s in after[:-1]],
                            fisher_sat_percentile=float(cfg["f_sat_percentile"]),
                            k_fixed=cfg["k_fixed"], grew=grew))
        except Exception as exc:  # recorded, never silenced
            trigger_paths[trig] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    rec2.flush()
    report["checks"]["trigger_paths"] = trigger_paths
    print("[11] growth-trigger paths: " + ", ".join(
        f"{k}={'grew' if v.get('grew') else ('ran' if v['ok'] else 'FAIL')}"
        for k, v in trigger_paths.items()))

    # 12. function-preserving insertion (validity check, not a result) ------
    params = fresh()
    probe = jax.random.normal(jax.random.PRNGKey(5), (16, cfg["n_in"]))
    out_before = model(params, probe)
    init_fn = resolve_init_fn("qr_init", jax.random.PRNGKey(1), params,
                              scale=cfg["qr_init_scale"])
    from noracl.growth.mlp_growth import add_neurons_with_fisher  # noqa: E402
    grown, _, _ = add_neurons_with_fisher(
        params, [[jnp.zeros_like(p) for p in params]],
        [[jnp.zeros_like(p) for p in params]], 0, 4, init_fn)
    out_after = model(grown, probe)
    fp_err = float(jnp.max(jnp.abs(out_after - out_before)))
    report["checks"]["function_preservation"] = {
        "max_abs_output_change": fp_err, "is_function_preserving": fp_err < 1e-5,
        "note": "zero fan-out insertion must not change the network function"}
    print(f"[12] function-preserving insertion: max|delta output| = {fp_err:.3e} "
          f"-> {'preserved' if fp_err < 1e-5 else 'NOT PRESERVED'}")

    (out / "smoke_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"\n{BANNER}")
    print(f"wrote {out/'smoke_report.json'}")
    print("No scientific arm was compared. No hypothesis was evaluated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
