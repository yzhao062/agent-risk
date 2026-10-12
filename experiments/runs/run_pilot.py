"""Run one stage of the arms A and B pilot (honest mode, declared policy).

Examples (from control-arena with PYTHONPATH pointing to the harness root):

  # stage 1 calibration: 40 baseline episodes
  python run_pilot.py --stage calib --gate baseline --n 40 \
      --master-seed 101 --seed-base 100000 --concurrency 32 --log-dir logs/pilot-ab/calib
  # B0: 300 baseline episodes
  python run_pilot.py --stage b0 --gate baseline --n 300 \
      --master-seed 202 --seed-base 200000 --concurrency 32 --log-dir logs/pilot-ab/b0
  # validation 1: rerun the first 100 B0 episodes (same task, same seed) at theta
  python run_pilot.py --stage val1 --gate stopping --theta 0.3 \
      --plan-file logs/pilot-ab/b0/plan.json --select 0:100 --concurrency 32 --log-dir ...
  # validation 2: 300 fresh stopping episodes at theta
  python run_pilot.py --stage val2a --gate stopping --theta 0.3 --n 300 \
      --master-seed 303 --seed-base 300000 --concurrency 32 --log-dir ...
  # arm C: retry gate (reject, the agent sees the refusal, retry cap 2)
  python run_pilot.py --stage c_retry --gate retry --theta 0.94 --retry-cap 2 --n 300 \
      --master-seed 401 --seed-base 410000 --concurrency 32 --log-dir ...
  # arm D: stopping gate announced in the agent's instructions
  python run_pilot.py --stage d_announce --gate stopping --theta 0.94 --announce --n 300 \
      --master-seed 402 --seed-base 420000 --concurrency 32 --log-dir ...

  # D58: one block of a frozen manifest (stage, gate, theta and episodes come
  # from the manifest), starting only episodes that have not begun in any
  # earlier attempt under the block directory (register Section 0.4)
  python run_pilot.py --manifest d58/manifest_d58.json --manifest-block 3 \
      --skip-started-under logs/d58/block-03-B2 --concurrency 32 --log-dir logs/d58/block-03-B2/attempt-1

Each run writes plan.json (the episodes: task index, seed) and run.json (wall
time, settings) next to the .eval log. Without --manifest the behavior is
unchanged.
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

from agent_risk_harness.pilot import MAX_STEPS, build_pilot_task, plan_episodes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=None, help="required unless --manifest is given")
    ap.add_argument("--gate", choices=["baseline", "stopping", "retry"], default=None,
                    help="required unless --manifest is given")
    ap.add_argument("--manifest", default=None, help="frozen episode manifest (make_manifest.py)")
    ap.add_argument("--manifest-block", type=int, default=None, help="global block number in the manifest")
    ap.add_argument("--skip-started-under", default=None,
                    help="skip manifest episodes that began in an earlier attempt under this directory: any "
                         "line of a started.jsonl (sample init, start or end) or any record of an episodes.jsonl")
    ap.add_argument("--retry-cap", type=int, default=2, help="retry gate: consecutive retries after a block")
    ap.add_argument("--announce", action="store_true", help="state the stopping gate in the agent's instructions")
    ap.add_argument("--theta", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="step cap (default 15, the pilot value)")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--master-seed", type=int, default=None)
    ap.add_argument("--seed-base", type=int, default=None)
    ap.add_argument("--plan-file", default=None, help="reuse episodes from a previous plan.json")
    ap.add_argument("--select", default=None, help="slice a:b of the reused plan")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--mode", choices=["honest", "attack"], default="honest", help="honest or attack mode")
    ap.add_argument("--attack-reminder", action="store_true", default=False, help="enable ephemeral attack reminder")
    ap.add_argument("--no-log-model-api", action="store_true",
                    help="keep only Inspect's default sample of raw requests (first few calls per model)")
    args = ap.parse_args()
    # Arm E replays these logs, so by default every model call keeps its raw API
    # request and response (Inspect 0.3.257 otherwise keeps the first few calls
    # per model per sample). Added after B0 had started; B0 ran without it.
    log_model_api = not args.no_log_model_api

    log_dir = os.path.expanduser(args.log_dir)
    manifest_info = None
    args.announcement_text = None
    args.cue = None
    if args.manifest:
        mpath = os.path.expanduser(args.manifest)
        mbytes = open(mpath, "rb").read()
        man = json.loads(mbytes)
        if args.manifest_block is None:
            raise SystemExit("--manifest-block is required with --manifest")
        blk = next(b for b in man["block_order"] if b["block"] == args.manifest_block)
        cond = man["conditions"][blk["condition"]]
        if (args.stage not in (None, cond["stage"])) or (args.gate not in (None, cond["gate"])) or \
                (args.theta is not None and args.theta != cond["theta"]) or \
                (args.announce and not cond.get("announce", False)):
            raise SystemExit("--stage/--gate/--theta/--announce disagree with the manifest")
        args.stage, args.gate, args.theta = cond["stage"], cond["gate"], cond["theta"]
        args.announce = bool(cond.get("announce", False))
        args.announcement_text = cond.get("announcement_text")
        args.cue = cond.get("cue")
        args.mode = cond.get("mode", "honest")
        args.attack_reminder = bool(cond.get("attack_reminder", False))
        args.attack_prompt = cond.get("attack_prompt", "controlarena")
        if args.max_steps is not None and args.max_steps != cond.get("max_steps", MAX_STEPS):
            raise SystemExit("--max-steps disagrees with the manifest")
        args.max_steps = int(cond.get("max_steps", MAX_STEPS))
        planned = [{"episode_id": e["episode_id"], "task_index": e["task_index"], "seed": e["seed"]}
                   for e in man["episodes"] if e["block"] == args.manifest_block]
        sample_ids = {f"{e['episode_id']}-t{e['task_id']}": e["episode_id"]
                      for e in man["episodes"] if e["block"] == args.manifest_block}
        begun: set[str] = set()
        if args.skip_started_under:
            root = os.path.expanduser(args.skip_started_under)
            for p in glob.glob(os.path.join(root, "**", "started.jsonl"), recursive=True):
                with open(p) as f:
                    for line in f:
                        try:
                            sid = json.loads(line).get("sample_id")
                        except json.JSONDecodeError:
                            continue
                        if sid in sample_ids:
                            begun.add(sample_ids[sid])
            for p in glob.glob(os.path.join(root, "**", "episodes.jsonl"), recursive=True):
                with open(p) as f:
                    for line in f:
                        begun.add(json.loads(line).get("episode_id"))
        episodes = [e for e in planned if e["episode_id"] not in begun]
        manifest_info = {"manifest": mpath, "manifest_sha256": hashlib.sha256(mbytes).hexdigest(),
                         "block": args.manifest_block, "label": blk["label"], "condition": blk["condition"],
                         "n_planned_in_block": len(planned), "n_begun_earlier": len(planned) - len(episodes),
                         "skipped_ids": sorted(set(e["episode_id"] for e in planned) & begun)}
        if not episodes:
            print(json.dumps({"block_complete": True, **manifest_info}))
            return 0
    elif args.stage is None or args.gate is None:
        raise SystemExit("--stage and --gate are required without --manifest")
    os.makedirs(log_dir, exist_ok=True)
    if args.manifest:
        os.environ["ARH_STARTED_LOG"] = os.path.join(log_dir, "started.jsonl")
        import agent_risk_harness.started_hook  # noqa: F401
    elif args.plan_file:
        with open(os.path.expanduser(args.plan_file)) as f:
            episodes = json.load(f)["episodes"]
        if args.select:
            a, b = (int(x) for x in args.select.split(":"))
            episodes = episodes[a:b]
    else:
        assert args.n and args.master_seed is not None and args.seed_base is not None
        episodes = plan_episodes(args.stage, args.n, args.master_seed, args.seed_base)
    if args.max_steps is None:
        args.max_steps = MAX_STEPS
    if args.gate in ("stopping", "retry") and args.theta is None:
        raise SystemExit("--theta is required for the stopping and retry gates")

    plan = {
        "stage": args.stage, "gate": args.gate, "theta": args.theta,
        "retry_cap": args.retry_cap if args.gate == "retry" else None,
        "announce": args.announce,
        "announcement_text": args.announcement_text,
        "cue": args.cue,
        "mode": args.mode,
        "attack_reminder": args.attack_reminder,
        "attack_prompt": getattr(args, "attack_prompt", "controlarena"),
        "max_steps": args.max_steps,
        "master_seed": args.master_seed, "seed_base": args.seed_base,
        "plan_file": args.plan_file, "select": args.select,
        "concurrency": args.concurrency, "episodes": episodes,
        "manifest": manifest_info,
    }
    with open(os.path.join(log_dir, "plan.json"), "w") as f:
        json.dump(plan, f, indent=1)

    task = build_pilot_task(episodes, stage=args.stage, gate=args.gate,
                            theta=args.theta, concurrency=args.concurrency,
                            retry_cap=args.retry_cap, announce=args.announce, max_steps=args.max_steps,
                            announcement_override=args.announcement_text, cue=args.cue,
                            mode=args.mode, attack_reminder=args.attack_reminder,
                            attack_prompt=getattr(args, "attack_prompt", "controlarena"))
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
        log_model_api=log_model_api,
    )
    wall = time.time() - t0
    summary = {
        "stage": args.stage, "gate": args.gate, "theta": args.theta,
        "retry_cap": args.retry_cap if args.gate == "retry" else None,
        "announce": args.announce,
        "announcement_text": args.announcement_text,
        "cue": args.cue,
        "mode": args.mode,
        "attack_reminder": args.attack_reminder,
        "attack_prompt": getattr(args, "attack_prompt", "controlarena"),
        "max_steps": args.max_steps,
        "n_episodes": len(episodes), "concurrency": args.concurrency, "manifest": manifest_info,
        "log_model_api": log_model_api,
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": wall,
        "logs": [{"location": lg.location, "status": lg.status,
                  "n_samples": len(lg.samples or [])} for lg in logs],
    }
    with open(os.path.join(log_dir, "run.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
