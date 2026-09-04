#!/usr/bin/env python3
"""Fetch the official NORACL implementation at the pinned revision.

Clones `karthik-charan/NORACL` into `third_party/noracl` (gitignored) and
verifies the checked-out SHA against the pin frozen in the preregistration. The
official code is never modified — instrumentation wraps it at runtime.

This script does not train anything.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modular_consolidation.native import NORACL_PIN  # noqa: E402

DEST = ROOT / "third_party" / "noracl"


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(DEST))
    ap.add_argument("--force", action="store_true", help="re-clone if present")
    args = ap.parse_args()

    dest = pathlib.Path(args.dest)
    if dest.exists() and args.force:
        shutil.rmtree(dest)

    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"cloning {NORACL_PIN['repo']} -> {dest}")
        run(["git", "clone", "--quiet", NORACL_PIN["repo"], str(dest)])
        run(["git", "-C", str(dest), "checkout", "--quiet", NORACL_PIN["sha"]])

    sha = run(["git", "-C", str(dest), "rev-parse", "HEAD"]).stdout.strip()
    dirty = run(["git", "-C", str(dest), "status", "--porcelain"]).stdout.strip()

    print(f"pinned sha : {NORACL_PIN['sha']}")
    print(f"checked out: {sha}")
    print(f"clean      : {not dirty}")
    if sha != NORACL_PIN["sha"]:
        print("MISMATCH — refusing to proceed; the official revision is not the pinned one.")
        return 2
    if dirty:
        print("MODIFIED — the official checkout must remain unmodified.")
        return 3

    for p in ("train.py", "configs/bsmnist_2l_noracl.yaml", "noracl/core/growth.py",
              "noracl/core/init.py", "noracl/training/loop.py",
              "results/paper/bsmnist_2l_noracl_s0/per_task.csv"):
        ok = (dest / p).exists()
        print(f"  {'ok ' if ok else 'MISSING'} {p}")
        if not ok:
            return 4

    print("\nofficial NORACL checkout verified at the pinned revision.")
    print("Run `python scripts/m6_smoke.py --help` for the score-free mechanical check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
