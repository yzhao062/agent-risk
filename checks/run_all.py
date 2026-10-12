#!/usr/bin/env python3
"""Run all finite-instance checks and exit non-zero if any fails."""

import os
import subprocess
import sys
import time
from pathlib import Path

CHECKS_DIR = Path(__file__).resolve().parent
REPO_ROOT = CHECKS_DIR.parent

SCRIPTS = [
    "verify_foundation.py",
    "verify_diagnostic.py",
    "verify_replay.py",
    "verify_horizon.py",
    "verify_supported.py",
    "verify_unsupported.py",
    "verify_coupling.py",
    "verify_eval_awareness.py",
    "verify_composition_replacement.py",
    "verify_budget.py",
    "verify_budget_limits.py",
    "verify_restoration.py",
    "verify_budget_sufficiency.py",
    "verify_budget_strata.py",
    "verify_snapshot_design.py",
]


def main():
    failed = []
    total_start = time.time()
    for script_name in SCRIPTS:
        script_path = CHECKS_DIR / script_name
        if not script_path.is_file():
            print(f"Error: {script_name} not found at {script_path}", file=sys.stderr)
            failed.append(script_name)
            continue

        print(f"=== Running {script_name} ===", flush=True)
        t0 = time.time()
        res = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(REPO_ROOT),
        )
        dt = time.time() - t0
        if res.returncode != 0:
            print(f"FAILED: {script_name} (exit code {res.returncode}, {dt:.1f}s)", flush=True)
            failed.append(script_name)
        else:
            print(f"PASSED: {script_name} ({dt:.1f}s)", flush=True)
        print(flush=True)

    total_dt = time.time() - total_start
    print(f"=== Summary ({total_dt:.1f}s) ===")
    if failed:
        print(f"Failed {len(failed)} of {len(SCRIPTS)} checks: {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"All {len(SCRIPTS)} checks passed successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
