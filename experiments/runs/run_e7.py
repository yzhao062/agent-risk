"""Run one block of the E7 manifest (pre-registration docs/agent-risk-modeling/E7-PREREG-2026-10-06.md).

A block is one eval holding base and live (gates 1 and 2) of one 50-unit chunk and cont (gates 1
and 2) of the previous chunk of the same host (make_manifest_e7.py).

  python ~/harness/run_e7.py --manifest ~/harness/e7/manifest_e7.json --manifest-block 7 \
      --logroot ~/logs/e7 --skip-started-under ~/logs/e7/h1-007 --concurrency 32 \
      --log-dir ~/logs/e7/h1-007/attempt-1

Cont units (blocks k >= 2) as E1's run_e1.py: the block refuses to start (exit 3) unless its source
block (k - 1) has a finished attempt (run.json). The units file <log-dir>/units.json is built from the
source block's .eval files (agent_risk_harness/e7_units.py). A planned cont episode runs only if its
gate blocked its base episode, the base episode has a committed snapshot at that step, the logged
prefix renders to the logged step-b request (prefix check) and the snapshot image is present;
otherwise it is not started and run.json lists it with the reason (an agreement unit, where the gate
never blocks, needs no continuation; every other reason makes the continuing outcome unknown in the
analysis). Each started cont episode gets a compose file (the pilot compose with the snapshot image)
and its source digest from the image label.

Resume and never-replace rule (as D58, D66 and E1): every sample that begins is appended to
<log-dir>/started.jsonl by the started hook; a manifest episode that appears in any started.jsonl or
episodes*.jsonl under --skip-started-under is never started again. Every run writes plan.json before
and run.json after the eval, including a run with nothing left to start.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone

from inspect_ai import eval as inspect_eval

from agent_risk_harness.e7 import build_e7_block_task
from agent_risk_harness.e7_units import block_eval_paths, build_units_doc
from snapshot_tools import compose_text


def image_info(tag: str) -> dict | None:
    """{'id', 'size', 'digest_label'} of a local E7 snapshot image, or None if absent."""
    r = subprocess.run(["docker", "image", "inspect", tag, "--format", "{{json .}}"], capture_output=True, text=True,
                       timeout=120)
    if r.returncode != 0:
        return None
    try:
        d = json.loads(r.stdout.strip())
    except json.JSONDecodeError:
        return None
    labels = ((d.get("Config") or {}).get("Labels")) or {}
    return {"id": d.get("Id"), "size": d.get("Size"), "digest_label": labels.get("arh.e7.digest"),
            "unit_label": labels.get("arh.e7.unit")}


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
    for p in glob.glob(os.path.join(root, "**", "episodes*.jsonl"), recursive=True):
        with open(p) as f:
            for line in f:
                try:
                    begun.add(json.loads(line).get("episode_id"))
                except json.JSONDecodeError:
                    continue
    return begun


def planned_items(man: dict, block: int) -> list[dict]:
    out = []
    for e in sorted((e for e in man["episodes"] if int(e["block"]) == int(block)),
                    key=lambda e: int(e["position_in_block"])):
        out.append({"episode_id": e["episode_id"], "unit_id": e["unit_id"], "base_unit_id": e["base_unit_id"],
                    "task_index": e["task_index"], "task_id": e["task_id"], "seed": e["seed"], "stage": e["stage"],
                    "condition": e["condition"], "gate": e["gate"], "base_episode_id": e["base_episode_id"]})
    return out


def write_compose(directory: str, unit_id: str, tag: str) -> str:
    os.makedirs(directory, exist_ok=True)
    path = os.path.abspath(os.path.join(directory, f"compose-{unit_id}.yml"))
    with open(path, "w") as f:
        f.write(compose_text(tag))
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--manifest-block", type=int, required=True)
    ap.add_argument("--logroot", required=True)
    ap.add_argument("--skip-started-under", default=None)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--no-log-model-api", action="store_true")
    args = ap.parse_args()

    mpath = os.path.expanduser(args.manifest)
    mbytes = open(mpath, "rb").read()
    man = json.loads(mbytes)
    lams = tuple(float(x) for x in man["lams"])
    blk = next(b for b in man["block_order"] if int(b["block"]) == args.manifest_block)
    stage = f"{man['prefix']}{blk['label']}"
    planned = planned_items(man, blk["block"])
    sample_ids = {f"{p['episode_id']}-t{p['task_id']}": p["episode_id"] for p in planned}
    log_dir = os.path.expanduser(args.log_dir)
    logroot = os.path.expanduser(args.logroot)
    os.makedirs(log_dir, exist_ok=True)

    cont_planned = [p for p in planned if p["condition"] == "cont"]
    units_doc = None
    units_path = os.path.join(log_dir, "units.json")
    if cont_planned:
        src = os.path.join(logroot, blk["source_label"])
        if not glob.glob(os.path.join(src, "attempt-*", "run.json")):
            print(json.dumps({"refused": f"source block {blk['source_label']} has no finished attempt"}))
            return 3
        units_doc = build_units_doc(block_eval_paths(src), cont_planned, lams, man["retry_cap"], blk["source_label"])
        with open(units_path, "w") as f:
            json.dump(units_doc, f)

    begun = begun_under(os.path.expanduser(args.skip_started_under), sample_ids) if args.skip_started_under else set()
    units_by_id = {u["unit_id"]: u for u in (units_doc or {}).get("units", [])}
    skipped_cont = {s["unit_id"]: s for s in (units_doc or {}).get("skipped", [])}
    not_run: list[dict] = []
    compose_files: dict[str, str] = {}
    source_digests: dict[str, str] = {}
    todo = []
    for p in planned:
        if p["episode_id"] in begun:
            continue
        if p["condition"] == "cont":
            u = units_by_id.get(p["unit_id"])
            if u is None:
                sk = skipped_cont.get(p["unit_id"]) or {"reason": "not prepared"}
                not_run.append({"episode_id": p["episode_id"], "unit_id": p["unit_id"], "reason": sk["reason"],
                                "agreement": bool(sk.get("agreement"))})
                continue
            if not u["prefix_check"]["ok"]:
                not_run.append({"episode_id": p["episode_id"], "unit_id": p["unit_id"], "reason": "prefix check failed",
                                "agreement": False})
                continue
            tag = u["snapshot"]["tag"]
            info = image_info(tag)
            if info is None:
                not_run.append({"episode_id": p["episode_id"], "unit_id": p["unit_id"], "reason": "snapshot image absent",
                                "agreement": False})
                continue
            if info.get("digest_label") != u["snapshot"]["digest"]:
                not_run.append({"episode_id": p["episode_id"], "unit_id": p["unit_id"],
                                "reason": "image digest label differs from the snapshot record", "agreement": False})
                continue
            compose_files[p["unit_id"]] = write_compose(os.path.join(log_dir, "compose"), p["unit_id"], tag)
            source_digests[p["unit_id"]] = u["snapshot"]["digest"]
        todo.append(p)

    os.environ["ARH_STARTED_LOG"] = os.path.join(log_dir, "started.jsonl")
    import agent_risk_harness.started_hook  # noqa: F401  (registers the hook)

    n_by_cond = {c: sum(1 for p in todo if p["condition"] == c) for c in man["conditions"]}
    info = {"manifest": mpath, "manifest_sha256": hashlib.sha256(mbytes).hexdigest(), "block": blk,
            "n_planned_in_block": len(planned), "skipped_begun": sorted(set(p["episode_id"] for p in planned) & begun),
            "cont_not_run": not_run, "n_to_run": len(todo), "n_to_run_by_condition": n_by_cond,
            "units_file": units_path if units_doc is not None else None,
            "units_file_sha256": (hashlib.sha256(open(units_path, "rb").read()).hexdigest()
                                  if units_doc is not None else None)}
    with open(os.path.join(log_dir, "plan.json"), "w") as f:
        json.dump({**info, "episodes": todo, "compose_files": compose_files, "source_digests": source_digests,
                   "concurrency": args.concurrency}, f, indent=1)
    t0 = time.time()
    started = datetime.now(timezone.utc).isoformat()
    logs = []
    if todo:
        task = build_e7_block_task([(p, p["condition"]) for p in todo], stage,
                                   units_file=units_path if units_doc is not None else None,
                                   compose_files=compose_files, source_digests=source_digests,
                                   concurrency=args.concurrency, lams=lams, retry_cap=man["retry_cap"],
                                   floor=float(man["floor"]))
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
    print(json.dumps({k: v for k, v in summary.items() if k not in ("block", "cont_not_run")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
