"""E5 manifests (E5-PREREG-2026-10-06.md): main and smoke.

main   E3's 1,200 units (u = 50 k + j: task j, seed index k, episode seed 830000 + u).
       Member C (live retry gate, arm C's rule, cap 2, gate (lam, pi)) for the units
       E3 did not run it on, k = 8..23 (units 400..1,199; ids e5c-<uuuu>), with E3's
       task and seed. Member R (continuation) for every unit of the units file (B's
       first block exists; ids e5r-<uuuu>). Eight blocks: block i (1..8) holds member
       C of seed indices 6 + 2i and 7 + 2i, member R of the units with those seed
       indices, and member R of the units with seed index i - 1 (whose live C ran in
       E3), so every block mixes live and continuation episodes. Episode order within
       a block is fixed here (random.Random(960 + i)).
smoke  Not analysed. Continuations restored from E3 pilot pass-all records (member P,
       seeds 810000 + u) and live retry episodes on pilot seeds, in two blocks.

  python e5/e5_make_manifest.py --kind main --units e5/e5_units.json --lam 11.186319 --pi 0 --out e5/manifest_e5.json
  python e5/e5_make_manifest.py --kind smoke --units e5/e5_units_smoke.json --lam 11.186319 --pi 0 --out e5/manifest_e5_smoke.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random

TAU2_COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"
N_TASKS = 50
N_SEED_IDX = 24
E3_C_K = 8                      # E3 ran member C on seed indices 0..7
N_BLOCKS = 8


def gate(lam: float, pi: float) -> dict:
    return {"kind": "retry", "lam": lam, "pi": pi, "retry_cap": 2}


def block_of_k_live(k: int) -> int:
    return (k - E3_C_K) // 2 + 1


def block_of_k_cont(k: int) -> int:
    return k + 1 if k < E3_C_K else block_of_k_live(k)


def main_manifest(units: dict, lam: float, pi: float, units_sha: str) -> dict:
    g = gate(lam, pi)
    eps = []
    for k in range(E3_C_K, N_SEED_IDX):
        for j in range(N_TASKS):
            u = N_TASKS * k + j
            eps.append({"episode_id": f"e5c-{u:04d}", "member": "C", "unit": u, "task_index": j, "task_id": str(j),
                        "seed_index": k, "seed": 830000 + u, "block": block_of_k_live(k), "gate": g})
    for x in units["units"]:
        u, k = int(x["unit"]), int(x["seed_index"])
        assert x["seed"] == 830000 + u and x["task_index"] == u % N_TASKS and k == u // N_TASKS
        eps.append({"episode_id": f"e5r-{u:04d}", "member": "R", "unit": u, "task_index": x["task_index"],
                    "task_id": x["task_id"], "seed_index": k, "seed": x["seed"], "b": x["b"],
                    "source_episode_id": x["source_episode_id"], "block": block_of_k_cont(k), "gate": g})
    blocks = []
    out = []
    for i in range(1, N_BLOCKS + 1):
        be = sorted([e for e in eps if e["block"] == i], key=lambda e: e["episode_id"])
        random.Random(960 + i).shuffle(be)
        for pos, e in enumerate(be):
            e["position"] = pos
        out.extend(be)
        blocks.append({"block": i, "label": f"block-{i:02d}-h1", "host": "h1", "stage": "main", "lam": lam, "pi": pi,
                       "n_live": sum(1 for e in be if e["member"] == "C"), "n_cont": sum(1 for e in be if e["member"] == "R")})
    return {"experiment": "E5", "kind": "main", "tau2_commit": TAU2_COMMIT, "domain": "airline", "task_split": "base",
            "lam": lam, "pi": pi, "units_file_sha256": units_sha, "e3_manifest_sha256": units["source"]["manifest_sha256"],
            "block_order": blocks, "episodes": out}


def smoke_manifest(units: dict, lam: float, pi: float, units_sha: str) -> dict:
    g = gate(lam, pi)
    us = units["units"]
    eps = []
    for n, x in enumerate(us):
        eps.append({"episode_id": f"e5sr-{int(x['unit']):03d}", "member": "R", "unit": int(x["unit"]),
                    "task_index": x["task_index"], "task_id": x["task_id"], "seed_index": x.get("seed_index"),
                    "seed": x["seed"], "b": x["b"], "source_episode_id": x["source_episode_id"],
                    "block": 1 if n < len(us) - 1 else 2, "gate": g})
    for n, x in enumerate(us[:3]):
        eps.append({"episode_id": f"e5sc-{int(x['unit']):03d}", "member": "C", "unit": int(x["unit"]),
                    "task_index": x["task_index"], "task_id": x["task_id"], "seed_index": x.get("seed_index"),
                    "seed": x["seed"], "block": 1 if n < 2 else 2, "gate": g})
    blocks = [{"block": 1, "label": "block-01-S", "host": "h1", "stage": "smoke", "lam": lam, "pi": pi},
              {"block": 2, "label": "block-02-S", "host": "h1", "stage": "smoke", "lam": lam, "pi": pi}]
    for i, e in enumerate(eps):
        e["position"] = i
    return {"experiment": "E5", "kind": "smoke", "tau2_commit": TAU2_COMMIT, "domain": "airline", "task_split": "base",
            "lam": lam, "pi": pi, "units_file_sha256": units_sha, "block_order": blocks, "episodes": eps}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("main", "smoke"), required=True)
    ap.add_argument("--units", required=True)
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--pi", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ub = open(a.units, "rb").read()
    units = json.loads(ub)
    sha = hashlib.sha256(ub).hexdigest()
    man = main_manifest(units, a.lam, a.pi, sha) if a.kind == "main" else smoke_manifest(units, a.lam, a.pi, sha)
    with open(a.out, "w") as f:
        json.dump(man, f, indent=1)
    print(a.out, len(man["episodes"]), "episodes", len(man["block_order"]), "blocks",
          {b["label"]: (b.get("n_live"), b.get("n_cont")) for b in man["block_order"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
