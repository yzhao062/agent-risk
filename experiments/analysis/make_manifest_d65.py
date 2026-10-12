"""Write the D65 long-horizon baseline manifests (register PLAN-RECONCILE-2026-10-04.md,
Section 0.2, D65).

One pass-all baseline log at a step cap K (40, or the largest cap the
feasibility smoke allows), n = 600 episodes drawn once with the function
run_pilot.py uses for --n/--master-seed/--seed-base
(agent_risk_harness.pilot.plan_episodes('d65', 600, 701, 710000)): episode ids
d65-0000 .. d65-0599, seeds 710000 + i. The log is split across two VMs:
episodes 0..299 run on <HOST-1> (blocks block-01 .. block-03), episodes
300..599 on <HOST-2> (blocks block-04 .. block-06), 100 per block. Each
VM gets its own manifest (block numbers 1..3 within it, global labels kept), so
the unchanged run_replication.sh runs exactly its VM's blocks; the combined
manifest lists all 600 for the analysis.

Run on a VM (ControlArena importable, for task ids):
  cd ~/control-arena && PYTHONPATH=~/harness .venv/bin/python ~/harness/make_manifest_d65.py \
      --max-steps 40 --outdir ~/harness/d65
Smoke (5 episodes, not part of the log): ... make_manifest_d65.py --smoke --max-steps 40 --outdir ~/harness/d65
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from datetime import datetime, timezone

from agent_risk_harness.pilot import N_TASKS, base_task_dataset, plan_episodes


def write(path: str, obj: dict) -> str:
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-steps", type=int, required=True)
    ap.add_argument("--smoke", action="store_true", help="5 episodes, stage smoke65, master seed 799, seeds 799000+")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    base = base_task_dataset()
    assert len(base) == N_TASKS == 257
    if args.smoke:
        stage, n, master, seed_base, block = "smoke65", 5, 799, 799000, 5
        vms = {"smoke": (0, 5)}
    else:
        stage, n, master, seed_base, block = "d65", 600, 701, 710000, 100
        vms = {"vm2": (0, 300), "vm3": (300, 600)}
    draws = plan_episodes(stage, n, master, seed_base)
    rng = random.Random(master)
    assert [d["task_index"] for d in draws] == [rng.randrange(N_TASKS) for _ in range(n)]
    cond = {"stage": stage, "gate": "baseline", "theta": None, "announce": False, "max_steps": args.max_steps}
    common = {
        "registered_in": "docs/agent-risk-modeling/PLAN-RECONCILE-2026-10-04.md, Section 0.2, D65",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "draw_function": f"agent_risk_harness.pilot.plan_episodes('{stage}', {n}, {master}, {seed_base})",
        "conditions": {"B": cond},
        "n_total": n, "master_seed": master, "seed_base": seed_base, "block_size": block,
        "max_steps": args.max_steps,
        "fixed_settings": "as B1: models, prompts, 400,000-token limit, checker v1.1, VLLM_BATCH_INVARIANT=1, "
                          "prefix caching off, concurrency 32, full raw-request logging; step cap max_steps",
    }
    all_eps, all_blocks, out = [], [], {}
    g = 0
    for vm, (a, b) in vms.items():
        eps, order = [], []
        for local, start in enumerate(range(a, b, block), start=1):
            g += 1
            label = f"block-{g:02d}-B"
            order.append({"block": local, "global_block": g, "label": label, "condition": "B", "vm": vm,
                          "n": min(block, b - start), "first_episode_id": f"{stage}-{start:04d}",
                          "last_episode_id": f"{stage}-{min(start + block, b) - 1:04d}"})
            for i in range(start, min(start + block, b)):
                d = draws[i]
                eps.append({"condition": "B", "stage": stage, "gate": "baseline", "theta": None,
                            "max_steps": args.max_steps, "vm": vm, "block": local, "global_block": g,
                            "position_in_block": i - start, "episode_id": f"{stage}-{i:04d}",
                            "seed": int(d["seed"]), "task_index": int(d["task_index"]),
                            "task_id": str(base[int(d["task_index"])].id)})
        name = f"manifest_{stage}_{vm}.json" if not args.smoke else f"manifest_{stage}.json"
        m = {"name": f"D65 long-horizon baseline, {vm}" if not args.smoke else "D65 feasibility smoke (not part of the log)",
             **common, "vm": vm, "block_order": order, "episodes": eps}
        out[name] = write(os.path.join(args.outdir, name), m)
        all_eps += eps
        all_blocks += order
    if not args.smoke:
        m = {"name": "D65 long-horizon baseline, combined (analysis)", **common, "vm": "both",
             "block_order": all_blocks, "episodes": all_eps}
        out[f"manifest_{stage}.json"] = write(os.path.join(args.outdir, f"manifest_{stage}.json"), m)
    print(json.dumps({"outdir": args.outdir, "sha256": out, "n_episodes": len(all_eps),
                      "distinct_tasks": len({e["task_index"] for e in all_eps})}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
