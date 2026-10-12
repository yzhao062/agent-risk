"""Write the E7 manifest (pre-registration docs/agent-risk-modeling/E7-PREREG-2026-10-06.md).

Units. n task draws with the function run_pilot.py uses
(agent_risk_harness.pilot.plan_episodes(prefix, n, master, seed_base)): unit i has task index
random.Random(master)'s i-th randrange(257) and episode seed seed_base + i. Unit ids
f"{prefix}-u{i:05d}"; episode ids f"{prefix}{code}-{i:05d}" with code b (base), v1 and v2 (live
retry gates 1 and 2), c1 and c2 (cont: the targeted execution of gate 1 and 2). A cont episode
runs only if its gate blocks the base log at some step and a snapshot exists at that step; it is
planned for every unit so that its id, block and position are fixed now. A cont unit id is
f"{unit_id}-g{g}".

Blocks. Units are split into chunks of 50. Chunk c belongs to host ((c - 1) mod H) + 1 and is the
host's k-th chunk, k = (c - 1) // H + 1; a host runs its chunks as blocks in order, and every
unit's five conditions run on its host. Block k of a host is one eval holding base and live
(gates 1 and 2) of the host's chunk k and cont (gates 1 and 2) of its chunk k - 1, whose
snapshots block k - 1 committed; a last block holds only the cont of the last chunk. Within block
k, samples start in the condition order ROTATION rotated left by (k - 1) mod 3, each condition's
units in unit order (a unit's gate 1 episode, then its gate 2 episode). After a block's attempt
finishes, the snapshots of chunk k - 1 are deleted. Labels f"h{host}-{k:03d}"; global block
numbers follow host order.

Smoke (--smoke): 8 units, prefix s7, master 977, seeds 977000 + i, chunks of 4, one host, blocks
h1-001 to h1-003, gates (9.0, 6.0) so that continuations occur in a few units; not analysed.

Run where ControlArena is importable (the VM's ~/control-arena venv):
  python make_manifest_e7.py --out e7/manifest_e7.json
  python make_manifest_e7.py --smoke --out e7/manifest_e7_smoke.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = {"base": "b", "live1": "v1", "live2": "v2", "cont1": "c1", "cont2": "c2"}
ROTATION = ["base", "live", "cont"]


def sha256(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def build_manifest(smoke: bool) -> dict:
    from agent_risk_harness.e7 import E7_SCHEMA, FLOOR, LAMBDAS, RETRY_CAP
    from agent_risk_harness.pilot import N_TASKS, base_task_dataset, plan_episodes

    if smoke:
        prefix, n, master, seed_base, csize, hosts = "s7", 8, 977, 977000, 4, 1
        lams = (9.0, 6.0)
    else:
        prefix, n, master, seed_base, csize, hosts = "e7", 7200, 1307, 1300000, 50, 6
        lams = LAMBDAS
    n_chunks = n // csize
    assert n % csize == 0 and n_chunks % hosts == 0
    per_host = n_chunks // hosts
    base = base_task_dataset()
    assert len(base) == N_TASKS == 257
    draws = plan_episodes(prefix, n, master, seed_base)
    rng = random.Random(master)
    assert [d["task_index"] for d in draws] == [rng.randrange(N_TASKS) for _ in range(n)]
    units = []
    for i, d in enumerate(draws):
        c = i // csize + 1
        h = (c - 1) % hosts + 1
        k = (c - 1) // hosts + 1
        uid = f"{prefix}-u{i:05d}"
        units.append({"unit": i, "unit_id": uid, "task_index": int(d["task_index"]),
                      "task_id": str(base[int(d["task_index"])].id), "seed": int(d["seed"]), "chunk": c, "host": h,
                      "k": k, "episode_ids": {m: f"{prefix}{CODES[m]}-{i:05d}" for m in CODES},
                      "cont_unit_ids": {"1": f"{uid}-g1", "2": f"{uid}-g2"}})
    order, episodes = [], []
    blk = 0
    for h in range(1, hosts + 1):
        chunk_of_k = {k: [u for u in units if u["host"] == h and u["k"] == k] for k in range(1, per_host + 1)}
        for k in range(1, per_host + 2):
            blk += 1
            label = f"h{h}-{k:03d}"
            a_units = chunk_of_k.get(k, [])
            b_units = chunk_of_k.get(k - 1, []) if k >= 2 else []
            rot = (k - 1) % len(ROTATION)
            groups = ROTATION[rot:] + ROTATION[:rot]
            items = []
            for grp in groups:
                if grp == "base":
                    items += [(u, "base", None) for u in a_units]
                elif grp == "live":
                    items += [(u, "live", g) for u in a_units for g in (1, 2)]
                else:
                    items += [(u, "cont", g) for u in b_units for g in (1, 2)]
            order.append({"block": blk, "label": label, "host": h, "k": k,
                          "a_chunk": a_units[0]["chunk"] if a_units else None,
                          "b_chunk": b_units[0]["chunk"] if b_units else None, "n_planned": len(items),
                          "source_label": f"h{h}-{k - 1:03d}" if b_units else None,
                          "start_order": [x for x in groups if any(c == x for _, c, _ in items)]})
            for j, (u, cond, g) in enumerate(items):
                if cond == "base":
                    eid, uid, code = u["episode_ids"]["base"], u["unit_id"], "b"
                elif cond == "live":
                    eid, uid, code = u["episode_ids"][f"live{g}"], u["unit_id"], f"v{g}"
                else:
                    eid, uid, code = u["episode_ids"][f"cont{g}"], u["cont_unit_ids"][str(g)], f"c{g}"
                episodes.append({"episode_id": eid, "unit": u["unit"], "unit_id": uid, "base_unit_id": u["unit_id"],
                                 "condition": cond, "gate": g, "stage": prefix + code, "block": blk, "label": label,
                                 "chunk": u["chunk"], "host": h, "position_in_block": j,
                                 "task_index": u["task_index"], "task_id": u["task_id"], "seed": u["seed"],
                                 "base_episode_id": u["episode_ids"]["base"], "conditional": cond == "cont"})
    return {
        "name": "E7 two retry gates by targeted execution from a change-agnostic log, stronger agent"
                + (" (smoke, not part of the batch)" if smoke else ""),
        "schema": E7_SCHEMA,
        "registered_in": "docs/agent-risk-modeling/PLAN-RECONCILE-2026-10-04.md, D72 to D74",
        "preregistration": "docs/agent-risk-modeling/E7-PREREG-2026-10-06.md",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "draw_function": f"agent_risk_harness.pilot.plan_episodes('{prefix}', {n}, {master}, {seed_base})",
        "prefix": prefix, "n_units": n, "master_seed": master, "seed_base": seed_base, "chunk_size": csize,
        "hosts": hosts, "lams": [float(x) for x in lams], "floor": float(FLOOR), "retry_cap": RETRY_CAP,
        "conditions": ROTATION, "codes": CODES,
        "chunks_per_host": per_host,
        "block_structure": "block k of a host = base and live (gates 1 and 2) of the host's chunk k plus cont (gates 1 and 2) "
                           "of its chunk k-1 (units whose base episode a gate blocked and whose snapshot at the first "
                           "block exists); start order ROTATION rotated by (k-1) mod 3; chunk k-1's snapshots are deleted "
                           "when the block has a finished attempt",
        "fixed_settings": "as E2: Qwen3.5-27B-FP8 agent on default kernels, Qwen3-4B monitor batch-invariant, gate "
                          "score s = -log10(1 - E[N]/100), checker v1.1, 15-step cap, 400,000 tokens per sample, "
                          "concurrency 32, full raw-request logging; retry gates at lams, retry cap 2; snapshot rule: "
                          "before every step whose logged s exceeds the floor",
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
