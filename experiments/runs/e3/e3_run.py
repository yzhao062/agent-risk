"""Run one block of an E3 manifest (E3-PREREG-2026-10-05.md).

Every episode of the block (all members of each task-seed unit) runs in this one
process against the same two server processes, CONCURRENCY at a time, in manifest
order. Episodes that began in an earlier attempt under the block directory are
skipped (a "start" line in any started.jsonl, or any episodes.jsonl record): a
begun episode is never executed again (the D58 rule).

Before importing tau2 the runner sets LITELLM_LOCAL_MODEL_COST_MAP=True (litellm
otherwise fetches its price map from GitHub at import) and installs the in-process
egress guard; the host rule (e3_net_block.sh) is checked by e3_run.sh.

  ~/venv-tau2/bin/python ~/harness/e3/e3_run.py --manifest ~/harness/e3/manifest_e3_pilot.json \
      --manifest-block 1 --skip-started-under ~/logs/e3_pilot/block-01-P --concurrency 32 \
      --log-dir ~/logs/e3_pilot/block-01-P/attempt-1

Writes plan.json, started.jsonl, episodes.jsonl (one full record per episode) and,
when every episode has returned, run.json.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
import e3_netguard  # noqa: E402

e3_netguard.install()


def begun_ids(root: str) -> set[str]:
    begun: set[str] = set()
    for p in glob.glob(os.path.join(root, "**", "started.jsonl"), recursive=True):
        with open(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("event") == "start":
                    begun.add(r.get("episode_id"))
    for p in glob.glob(os.path.join(root, "**", "episodes.jsonl"), recursive=True):
        with open(p) as f:
            for line in f:
                try:
                    begun.add(json.loads(line).get("episode_id"))
                except json.JSONDecodeError:
                    continue
    return begun


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--manifest-block", type=int, required=True)
    ap.add_argument("--skip-started-under", default=None)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--agent-url", default=None)
    ap.add_argument("--monitor-url", default=None)
    args = ap.parse_args()

    mpath = os.path.expanduser(args.manifest)
    mbytes = open(mpath, "rb").read()
    man = json.loads(mbytes)
    if man.get("experiment") != "E3":
        raise SystemExit("not an E3 manifest")
    blk = next(b for b in man["block_order"] if b["block"] == args.manifest_block)
    planned = [e for e in man["episodes"] if e["block"] == args.manifest_block]
    begun = begun_ids(os.path.expanduser(args.skip_started_under)) if args.skip_started_under else set()
    todo = [e for e in planned if e["episode_id"] not in begun]
    info = {"manifest": mpath, "manifest_sha256": hashlib.sha256(mbytes).hexdigest(), "block": args.manifest_block,
            "label": blk["label"], "host": blk.get("host"), "stage": blk["stage"],
            "n_planned_in_block": len(planned), "n_begun_earlier": len(planned) - len(todo),
            "skipped_ids": sorted({e["episode_id"] for e in planned} & begun)}
    if not todo:
        print(json.dumps({"block_complete": True, **info}))
        return 0

    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="ERROR")      # tau2 logs every cancellation at WARNING
    import e3_episode
    import e3_llm
    import e3_monitor
    from tau2.domains.airline.environment import get_tasks
    e3_llm.install()
    tasks = {t.id: t for t in get_tasks(man.get("task_split", "base"))}
    agent_url = args.agent_url or e3_llm.AGENT_URL
    monitor_url = args.monitor_url or e3_monitor.MONITOR_URL

    log_dir = os.path.expanduser(args.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    plan = {"experiment": "E3", "block": info, "concurrency": args.concurrency, "episodes": todo,
            "agent_url": agent_url, "monitor_url": monitor_url}
    with open(os.path.join(log_dir, "plan.json"), "w") as f:
        json.dump(plan, f, indent=1)
    lock = threading.Lock()
    started_f = open(os.path.join(log_dir, "started.jsonl"), "a")
    eps_f = open(os.path.join(log_dir, "episodes.jsonl"), "a")

    def write(fh, obj):
        with lock:
            fh.write(json.dumps(obj) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    counts = {"done": 0, "scored": 0, "errors": 0}

    def one(e: dict) -> None:
        write(started_f, {"episode_id": e["episode_id"], "event": "start", "utc": utc()})
        rec = e3_episode.run_episode(tasks[e["task_id"]], int(e["seed"]), e["gate"], agent_url=agent_url,
                                     monitor_url=monitor_url)
        rec.update({"episode_id": e["episode_id"], "member": e["member"], "unit": e["unit"],
                    "task_index": e["task_index"], "seed_index": e.get("seed_index"), "block": e["block"],
                    "block_label": blk["label"], "host": blk.get("host"), "finished_utc": utc(),
                    "egress_attempts": list(e3_netguard.ATTEMPTS)})
        write(eps_f, rec)
        write(started_f, {"episode_id": e["episode_id"], "event": "end", "utc": utc(), "scored": rec.get("scored")})
        with lock:
            counts["done"] += 1
            counts["scored"] += int(bool(rec.get("scored")))
            counts["errors"] += int(not rec.get("scored"))

    t0 = time.time()
    started = utc()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(one, todo))
    summary = {"experiment": "E3", "block": info, "n_episodes": len(todo), "concurrency": args.concurrency,
               "started_utc": started, "finished_utc": utc(), "wall_seconds": time.time() - t0, **counts,
               "egress_attempts": list(e3_netguard.ATTEMPTS)}
    with open(os.path.join(log_dir, "run.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
