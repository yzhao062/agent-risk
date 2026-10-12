"""Write the E1 manifest (register PLAN-RECONCILE-2026-10-04.md, D72; pre-registration
docs/agent-risk-modeling/E1-PREREG-2026-10-05.md).

Units. n task draws with the function run_pilot.py uses
(agent_risk_harness.pilot.plan_episodes(prefix, n, master, seed_base)): unit i
has task index random.Random(master)'s i-th randrange(257) and episode seed
seed_base + i. Unit ids f"{prefix}-u{i:04d}"; episode ids f"{prefix}{code}-{i:04d}"
with code b (base), v (live), c (cont). The cont episode of a unit runs only if
its base episode deviates (a step scored above theta) and committed a snapshot;
it is planned for every unit so that its id, block and position are fixed now.

Blocks. Units are split into chunks of 30; chunks 1 to 20 (units 0 to 599) form
half "a" and chunks 21 to 40 half "b"; each half runs on its own VM, so every
unit's three conditions run on one host. A half runs 21 blocks in order. Block
k of a half is one eval holding base and live of the half's chunk k (k <= 20)
and cont of chunk k - 1 (k >= 2), whose snapshots block k - 1 committed. Within
block k, samples start in the condition order ROTATION rotated left by
(k - 1) mod 3, each condition's units in unit order. After a block's attempt
finishes, the snapshots of chunk k - 1 are deleted. Labels f"{half}{k:02d}";
global block numbers 1 to 42 (half a first).

Smoke (--smoke): 6 units, prefix s1, master seed 979, seeds 979000 + i, chunks
of 3, one half, blocks a01 to a03.

Mac (ControlArena importable for task ids; a docker shim on PATH):
  python make_manifest_e1.py --out e1/manifest_e1.json
  python make_manifest_e1.py --smoke --out e1/manifest_e1_smoke.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = {"base": "b", "live": "v", "cont": "c"}
ROTATION = ["base", "live", "cont"]


def sha256(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def build_manifest(smoke: bool) -> dict:
    from agent_risk_harness.e1 import E1_SCHEMA, RETRY_CAP, THETA
    from agent_risk_harness.pilot import N_TASKS, base_task_dataset, plan_episodes

    if smoke:
        prefix, n, master, seed_base, csize, n_halves = "s1", 6, 979, 979000, 3, 1
    else:
        prefix, n, master, seed_base, csize, n_halves = "e1", 1200, 901, 910000, 30, 2
    n_chunks = n // csize
    per_half = n_chunks // n_halves
    base = base_task_dataset()
    assert len(base) == N_TASKS == 257
    draws = plan_episodes(prefix, n, master, seed_base)
    rng = random.Random(master)
    assert [d["task_index"] for d in draws] == [rng.randrange(N_TASKS) for _ in range(n)]
    units = []
    for i, d in enumerate(draws):
        c = i // csize + 1
        units.append({"unit": i, "unit_id": f"{prefix}-u{i:04d}", "task_index": int(d["task_index"]),
                      "task_id": str(base[int(d["task_index"])].id), "seed": int(d["seed"]), "chunk": c,
                      "half": "ab"[(c - 1) // per_half],
                      "episode_ids": {m: f"{prefix}{CODES[m]}-{i:04d}" for m in ROTATION}})
    order, episodes = [], []
    blk = 0
    for h in range(n_halves):
        half = "ab"[h]
        chunks = list(range(h * per_half + 1, (h + 1) * per_half + 1))
        for k in range(1, per_half + 2):
            blk += 1
            label = f"{half}{k:02d}"
            a_chunk = chunks[k - 1] if k <= per_half else None
            b_chunk = chunks[k - 2] if k >= 2 else None
            rot = (k - 1) % len(ROTATION)
            items = []
            for cond in ROTATION[rot:] + ROTATION[:rot]:
                c = a_chunk if cond in ("base", "live") else b_chunk
                if c is not None:
                    items += [(u, cond) for u in units if u["chunk"] == c]
            order.append({"block": blk, "label": label, "half": half, "k": k, "a_chunk": a_chunk, "b_chunk": b_chunk,
                          "n_planned": len(items), "source_label": f"{half}{k - 1:02d}" if b_chunk is not None else None,
                          "start_order": [c for c in ROTATION[rot:] + ROTATION[:rot]
                                          if any(x == c for _, x in items)]})
            for j, (u, cond) in enumerate(items):
                episodes.append({"episode_id": u["episode_ids"][cond], "unit": u["unit"], "unit_id": u["unit_id"],
                                 "condition": cond, "stage": prefix + CODES[cond], "block": blk, "label": label,
                                 "chunk": u["chunk"], "half": half, "position_in_block": j,
                                 "task_index": u["task_index"], "task_id": u["task_id"], "seed": u["seed"],
                                 "base_episode_id": u["episode_ids"]["base"],
                                 "conditional": cond == "cont"})
    return {
        "name": "E1 retry gate by targeted execution" + (" (smoke, not part of the batch)" if smoke else ""),
        "schema": E1_SCHEMA,
        "registered_in": "docs/agent-risk-modeling/PLAN-RECONCILE-2026-10-04.md, D72",
        "preregistration": "docs/agent-risk-modeling/E1-PREREG-2026-10-05.md",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "draw_function": f"agent_risk_harness.pilot.plan_episodes('{prefix}', {n}, {master}, {seed_base})",
        "prefix": prefix, "n_units": n, "master_seed": master, "seed_base": seed_base, "chunk_size": csize,
        "theta": THETA, "retry_cap": RETRY_CAP, "conditions": ROTATION, "codes": CODES,
        "halves": {hv: sorted({u["chunk"] for u in units if u["half"] == hv}) for hv in "ab"[:n_halves]},
        "block_structure": "block k of a half = base and live of chunk k plus cont of chunk k-1 (units whose base "
                           "episode deviated and committed a snapshot); start order ROTATION rotated by (k-1) mod 3; "
                           "chunk k-1's snapshots are deleted when the block has a finished attempt",
        "fixed_settings": "as B1, arm C, D58, D64 and D66: models, prompts, monitor, checker v1.1, 15-step cap, "
                          "400,000 tokens per sample, VLLM_BATCH_INVARIANT=1, prefix caching off, concurrency 32, "
                          "full raw-request logging; retry gate theta 0.94, retry cap 2",
        "units": units,
        "block_order": order,
        "episodes": episodes,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    man = build_manifest(args.smoke)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(man, f, indent=1)
    print(json.dumps({"out": args.out, "sha256": sha256(args.out), "n_units": man["n_units"],
                      "n_episodes": len(man["episodes"]), "n_blocks": len(man["block_order"]),
                      "n_distinct_tasks": len({u["task_index"] for u in man["units"]}),
                      "first_blocks": [(o["label"], o["n_planned"], o["start_order"]) for o in man["block_order"][:3]]},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
