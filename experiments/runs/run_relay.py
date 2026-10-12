"""Run one block of the D66 relay manifest (register PLAN-RECONCILE-2026-10-04.md,
D66). A block is one eval holding the A part of one 16-unit chunk (relay, first
and second_clean) and the B part of the previous chunk of the same half
(second_snap and second_rep), in the manifest's order (make_manifest_relay.py).

  python ~/harness/run_relay.py --manifest ~/harness/d66/manifest_d66.json --manifest-block 7 \
      --logroot ~/logs/d66 --skip-started-under ~/logs/d66/a07 \
      --concurrency 32 --log-dir ~/logs/d66/a07/attempt-1

Resume and never-replace rule (as D58): every sample that begins is appended to
<log-dir>/started.jsonl by the started hook; a manifest episode that appears in
any started.jsonl or episodes.jsonl under --skip-started-under is never started
again, and its outcome is unknown if its record cannot be recovered.

B part: the block refuses to start (exit 3) unless its source block (the
previous block of the half, which ran the chunk's relay and first) has a
finished attempt (run.json). Units whose source image
arh-relay:<unit_id>-<source mode> is absent are not started and are listed in
run.json as missing_source_snapshot (outcome unknown). Each started B-part unit
gets a compose file under <log-dir>/compose/ and its source handoff digest from
the image label, which the start solver compares with the restored file tree.

Every run writes plan.json before and run.json after the eval, including a run
with nothing left to start, so the chain counts the block as done. Deleting the
source snapshots after a finished attempt is the chain's step
(snapshot_tools.py cleanup-block), not this script's.
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

from agent_risk_harness.relay import FROM_SNAPSHOT, build_relay_block_task, snapshot_tag
from snapshot_tools import image_info, write_compose


def begun_under(root: str, ids: dict[str, str]) -> set[str]:
    """Episode ids that began in any earlier attempt under root (sample ids map to episode ids)."""
    begun: set[str] = set()
    for p in glob.glob(os.path.join(root, "**", "started.jsonl"), recursive=True):
        with open(p) as f:
            for line in f:
                try:
                    sid = json.loads(line).get("sample_id")
                except json.JSONDecodeError:
                    continue
                if sid in ids:
                    begun.add(ids[sid])
    for p in glob.glob(os.path.join(root, "**", "episodes.jsonl"), recursive=True):
        with open(p) as f:
            for line in f:
                try:
                    begun.add(json.loads(line).get("episode_id"))
                except json.JSONDecodeError:
                    continue
    return begun


def planned_items(man: dict, block: int) -> list[dict]:
    """The block's episodes in manifest order, as relay unit dicts (one per sample)."""
    out = []
    for e in sorted((e for e in man["episodes"] if int(e["block"]) == int(block)),
                    key=lambda e: int(e["position_in_block"])):
        out.append({"episode_id": e["episode_id"], "unit_id": e["unit_id"], "task_index": e["task_index"],
                    "seed_a1": e["seed_a1"], "seed_a2": e["seed_a2"], "task_class": e["task_class"],
                    "task_id": e["task_id"], "stage": e["stage"], "mode": e["mode"], "part": e["part"]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--manifest-block", type=int, required=True)
    ap.add_argument("--logroot", required=True, help="log root holding every block directory (B-part source check)")
    ap.add_argument("--skip-started-under", default=None)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--no-log-model-api", action="store_true")
    args = ap.parse_args()

    mpath = os.path.expanduser(args.manifest)
    mbytes = open(mpath, "rb").read()
    man = json.loads(mbytes)
    blk = next(b for b in man["block_order"] if int(b["block"]) == args.manifest_block)
    stage = f"{man['prefix']}{blk['label']}"
    planned = planned_items(man, blk["block"])
    sample_ids = {f"{p['episode_id']}-t{p['task_id']}": p["episode_id"] for p in planned}
    log_dir = os.path.expanduser(args.log_dir)
    logroot = os.path.expanduser(args.logroot)

    if blk.get("b_chunk") is not None:
        src = os.path.join(logroot, blk["source_label"])
        if not glob.glob(os.path.join(src, "attempt-*", "run.json")):
            print(json.dumps({"refused": f"source block {blk['source_label']} has no finished attempt"}))
            return 3

    begun = begun_under(os.path.expanduser(args.skip_started_under), sample_ids) if args.skip_started_under else set()
    todo = [p for p in planned if p["episode_id"] not in begun]
    missing_src: list[str] = []
    compose_files: dict[str, str] = {}
    source_digests: dict[str, str] = {}
    keep = []
    for p in todo:
        mode = p["mode"]
        if mode in FROM_SNAPSHOT:
            tag = snapshot_tag(p["unit_id"], FROM_SNAPSHOT[mode])
            info = image_info(tag)
            if info is None:
                missing_src.append(p["episode_id"])
                continue
            # one compose file per (unit, source mode); a unit's second_snap and second_rep use different files
            compose_files[f"{p['unit_id']}|{mode}"] = write_compose(os.path.join(log_dir, "compose"), p["unit_id"],
                                                                     FROM_SNAPSHOT[mode])
            if info.get("digest_label"):
                source_digests[f"{p['unit_id']}|{mode}"] = info["digest_label"]
        keep.append(p)
    todo = keep

    os.makedirs(log_dir, exist_ok=True)
    os.environ["ARH_STARTED_LOG"] = os.path.join(log_dir, "started.jsonl")
    import agent_risk_harness.started_hook  # noqa: F401  (registers the hook)

    n_by_mode = {m: sum(1 for p in todo if p["mode"] == m) for m in man["modes"]}
    info = {"manifest": mpath, "manifest_sha256": hashlib.sha256(mbytes).hexdigest(), "block": blk,
            "n_planned_in_block": len(planned), "n_begun_earlier": len(planned) - len(todo) - len(missing_src),
            "skipped_ids": sorted(set(p["episode_id"] for p in planned) & begun),
            "missing_source_snapshot": missing_src, "n_to_run": len(todo), "n_to_run_by_mode": n_by_mode}
    with open(os.path.join(log_dir, "plan.json"), "w") as f:
        json.dump({**info, "episodes": todo, "compose_files": compose_files, "source_digests": source_digests,
                   "concurrency": args.concurrency}, f, indent=1)
    t0 = time.time()
    started = datetime.now(timezone.utc).isoformat()
    logs = []
    if todo:
        task = build_relay_block_task([(p, p["mode"]) for p in todo], stage, concurrency=args.concurrency,
                                      compose_files=compose_files, source_digests=source_digests)
        logs = inspect_eval(
            task, log_dir=log_dir, max_tasks=1, max_samples=args.concurrency, max_sandboxes=args.concurrency,
            max_subprocesses=max(64, 3 * args.concurrency), fail_on_error=False, log_level="warning",
            display="plain", log_model_api=not args.no_log_model_api,
        )
    summary = {**info, "stage": stage, "concurrency": args.concurrency,
               "log_model_api": not args.no_log_model_api, "started_utc": started,
               "finished_utc": datetime.now(timezone.utc).isoformat(), "wall_seconds": time.time() - t0,
               "logs": [{"location": lg.location, "status": lg.status, "n_samples": len(lg.samples or [])}
                        for lg in logs]}
    with open(os.path.join(log_dir, "run.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "block"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
