"""Run one block of an E5 manifest (E5-PREREG-2026-10-06.md).

Every episode of the block runs in this one process against the same two server
processes, CONCURRENCY at a time, in manifest order: member C through E3's frozen
``e3_episode.run_episode`` (a fresh live retry episode), member R through
``e5_episode.run_continuation`` (the live retry gate from state restored at the
first block, from the frozen units file). Episodes that began in an earlier attempt
under the block directory are skipped (a "start" line in any started.jsonl, or any
record): a begun episode is never executed again (the D58 rule).

Records go to separate files so that the prediction never opens a live record:
episodes_cont.jsonl (member R) and episodes_live.jsonl (member C); started.jsonl
carries start and end events (episode id, member, time; no outcome); run.json is
written when every episode has returned. The units file and the manifest must have
the sha256 the manifest and the chain name.

  ~/venv-tau2/bin/python ~/harness/e5/e5_run.py --manifest ~/harness/e5/manifest_e5.json \
      --units ~/harness/e5/e5_units.json --manifest-block 1 --skip-started-under ~/logs/e5/block-01-h1 \
      --concurrency 32 --log-dir ~/logs/e5/block-01-h1/attempt-1
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
HARNESS = os.path.dirname(HERE)
E3 = os.path.join(HARNESS, "e3")
for p in (E3, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
import e3_netguard  # noqa: E402

e3_netguard.install()

OUT_FILES = {"R": "episodes_cont.jsonl", "C": "episodes_live.jsonl"}


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
    for name in OUT_FILES.values():
        for p in glob.glob(os.path.join(root, "**", name), recursive=True):
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
    ap.add_argument("--units", required=True)
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
    if man.get("experiment") != "E5":
        raise SystemExit("not an E5 manifest")
    ubytes = open(os.path.expanduser(args.units), "rb").read()
    usha = hashlib.sha256(ubytes).hexdigest()
    if usha != man.get("units_file_sha256"):
        raise SystemExit(f"REFUSING: units file sha256 {usha} differs from the manifest's {man.get('units_file_sha256')}")
    units = {int(x["unit"]): x for x in json.loads(ubytes)["units"]}
    blk = next(b for b in man["block_order"] if b["block"] == args.manifest_block)
    planned = sorted([e for e in man["episodes"] if e["block"] == args.manifest_block], key=lambda e: e["position"])
    begun = begun_ids(os.path.expanduser(args.skip_started_under)) if args.skip_started_under else set()
    todo = [e for e in planned if e["episode_id"] not in begun]
    info = {"manifest": mpath, "manifest_sha256": hashlib.sha256(mbytes).hexdigest(), "units_sha256": usha,
            "block": args.manifest_block, "label": blk["label"], "host": blk.get("host"), "stage": blk["stage"],
            "n_planned_in_block": len(planned), "n_begun_earlier": len(planned) - len(todo),
            "skipped_ids": sorted({e["episode_id"] for e in planned} & begun)}
    if not todo:
        print(json.dumps({"block_complete": True, **info}))
        return 0

    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    import e3_episode
    import e3_llm
    import e3_monitor
    import e5_episode
    from tau2.domains.airline.environment import get_tasks
    e3_llm.install()
    tasks = {t.id: t for t in get_tasks(man.get("task_split", "base"))}
    agent_url = args.agent_url or e3_llm.AGENT_URL
    monitor_url = args.monitor_url or e3_monitor.MONITOR_URL

    log_dir = os.path.expanduser(args.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    plan = {"experiment": "E5", "block": info, "concurrency": args.concurrency,
            "episodes": [{k: e[k] for k in ("episode_id", "member", "unit", "task_id", "seed", "position")} for e in todo],
            "agent_url": agent_url, "monitor_url": monitor_url}
    with open(os.path.join(log_dir, "plan.json"), "w") as f:
        json.dump(plan, f, indent=1)
    lock = threading.Lock()
    started_f = open(os.path.join(log_dir, "started.jsonl"), "a")
    out_f = {m: open(os.path.join(log_dir, name), "a") for m, name in OUT_FILES.items()}

    def write(fh, obj):
        with lock:
            fh.write(json.dumps(obj) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    counts = {"done": 0, "scored": 0, "errors": 0, "R": 0, "C": 0}

    def one(e: dict) -> None:
        write(started_f, {"episode_id": e["episode_id"], "member": e["member"], "event": "start", "utc": utc()})
        task = tasks[e["task_id"]]
        if e["member"] == "R":
            u = units[int(e["unit"])]
            assert u["seed"] == e["seed"] and u["b"] == e["b"] and u["source_episode_id"] == e["source_episode_id"]
            rec = e5_episode.run_continuation(u, task, e["gate"], agent_url=agent_url, monitor_url=monitor_url)
        elif e["member"] == "C":
            rec = e3_episode.run_episode(task, int(e["seed"]), e["gate"], agent_url=agent_url, monitor_url=monitor_url)
        else:
            raise ValueError(e["member"])
        rec.update({"episode_id": e["episode_id"], "member": e["member"], "unit": e["unit"],
                    "task_index": e["task_index"], "seed_index": e.get("seed_index"), "block": e["block"],
                    "block_label": blk["label"], "host": blk.get("host"), "finished_utc": utc(),
                    "egress_attempts": list(e3_netguard.ATTEMPTS)})
        write(out_f[e["member"]], rec)
        write(started_f, {"episode_id": e["episode_id"], "member": e["member"], "event": "end", "utc": utc()})
        with lock:
            counts["done"] += 1
            counts["scored"] += int(bool(rec.get("scored")))
            counts["errors"] += int(not rec.get("scored"))
            counts[e["member"]] += 1

    t0 = time.time()
    started = utc()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(one, todo))
    summary = {"experiment": "E5", "block": info, "n_episodes": len(todo), "concurrency": args.concurrency,
               "started_utc": started, "finished_utc": utc(), "wall_seconds": time.time() - t0, **counts,
               "egress_attempts": list(e3_netguard.ATTEMPTS)}
    with open(os.path.join(log_dir, "run.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in ("n_episodes", "done", "R", "C", "wall_seconds")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
