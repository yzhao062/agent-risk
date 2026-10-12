"""Run one block of an E2 manifest (register D72; E2-PREREG-2026-10-05.md).

Like run_pilot.py --manifest (D58, D64), with the E2 task builder: one eval per
block holds the block's episodes in manifest order, each carrying its own gate
(pass-all or live stopping) from the manifest; the gate parameters (lam, pi)
come from the block entry. Episodes that began in an earlier attempt under the
block directory are skipped (any started.jsonl line naming the sample, or any
episodes.jsonl record), and a begun episode is never executed again.

  cd ~/control-arena && PYTHONPATH=~/harness .venv/bin/python ~/harness/e2_run.py \
      --manifest ~/harness/e2/manifest_e2_pilot.json --manifest-block 1 \
      --skip-started-under ~/logs/e2_pilot/block-01-P --concurrency 32 \
      --log-dir ~/logs/e2_pilot/block-01-P/attempt-1

Writes plan.json, started.jsonl, the .eval log and run.json in --log-dir.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import time
from datetime import datetime, timezone

from inspect_ai import eval as inspect_eval


def begun_ids(root: str, sample_ids: dict[str, str]) -> set[str]:
    begun: set[str] = set()
    for p in glob.glob(os.path.join(root, "**", "started.jsonl"), recursive=True):
        with open(p) as f:
            for line in f:
                try:
                    sid = json.loads(line).get("sample_id")
                except json.JSONDecodeError:
                    continue  # a line cut by an interruption
                if sid in sample_ids:
                    begun.add(sample_ids[sid])
    for p in glob.glob(os.path.join(root, "**", "episodes.jsonl"), recursive=True):
        with open(p) as f:
            for line in f:
                try:
                    begun.add(json.loads(line).get("episode_id"))
                except json.JSONDecodeError:
                    continue
    return begun


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--manifest-block", type=int, required=True)
    ap.add_argument("--skip-started-under", default=None)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--log-dir", required=True)
    args = ap.parse_args()

    mpath = os.path.expanduser(args.manifest)
    mbytes = open(mpath, "rb").read()
    man = json.loads(mbytes)
    if man.get("experiment") != "E2":
        raise SystemExit("not an E2 manifest")
    blk = next(b for b in man["block_order"] if b["block"] == args.manifest_block)
    lam, pi = float(blk["lam"]), float(blk["pi"])
    planned = [e for e in man["episodes"] if e["block"] == args.manifest_block]
    sample_ids = {f"{e['episode_id']}-t{e['task_id']}": e["episode_id"] for e in planned}
    begun = begun_ids(os.path.expanduser(args.skip_started_under), sample_ids) if args.skip_started_under else set()
    episodes = [{"episode_id": e["episode_id"], "task_index": e["task_index"], "seed": e["seed"],
                 "gate": e["gate"], "condition": e["condition"], "pair": e.get("pair")}
                for e in planned if e["episode_id"] not in begun]
    info = {"manifest": mpath, "manifest_sha256": hashlib.sha256(mbytes).hexdigest(),
            "block": args.manifest_block, "label": blk["label"], "host": blk.get("host"),
            "lam": lam, "pi": pi, "stage": blk["stage"],
            "n_planned_in_block": len(planned), "n_begun_earlier": len(planned) - len(episodes),
            "skipped_ids": sorted(set(e["episode_id"] for e in planned) & begun)}
    if not episodes:
        print(json.dumps({"block_complete": True, **info}))
        return 0

    log_dir = os.path.expanduser(args.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    os.environ["ARH_STARTED_LOG"] = os.path.join(log_dir, "started.jsonl")
    import agent_risk_harness.started_hook  # noqa: F401  (registers the hook)
    from agent_risk_harness.e2_pilot import build_e2_task
    from agent_risk_harness.pilot import MAX_STEPS

    max_steps = int(blk.get("max_steps", MAX_STEPS))
    plan = {"experiment": "E2", "block": info, "concurrency": args.concurrency, "max_steps": max_steps,
            "episodes": episodes}
    with open(os.path.join(log_dir, "plan.json"), "w") as f:
        json.dump(plan, f, indent=1)

    task = build_e2_task(episodes, stage=blk["stage"], lam=lam, pi=pi, concurrency=args.concurrency,
                         max_steps=max_steps, block=info)
    t0 = time.time()
    started = datetime.now(timezone.utc).isoformat()
    logs = inspect_eval(
        task,
        log_dir=log_dir,
        max_tasks=1,
        max_samples=args.concurrency,
        max_sandboxes=args.concurrency,
        max_subprocesses=max(64, 3 * args.concurrency),
        fail_on_error=False,
        log_level="warning",
        display="plain",
        log_model_api=True,
    )
    summary = {"experiment": "E2", "block": info, "n_episodes": len(episodes), "concurrency": args.concurrency,
               "max_steps": max_steps, "log_model_api": True,
               "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
               "wall_seconds": time.time() - t0,
               "logs": [{"location": lg.location, "status": lg.status, "n_samples": len(lg.samples or [])}
                        for lg in logs]}
    with open(os.path.join(log_dir, "run.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
