"""E3 manifests (E3-PREREG-2026-10-05.md): smoke, pilot, confirmatory.

Tasks are the 50 tau2-bench airline tasks of split "base" at tau2 v1.0.1, indexed
0..49 in the order of their ids "0".."49". A unit is (task index j, seed index k)
with one episode seed; its members share task and seed.

  smoke    2 blocks, not analysed. Block 1: tasks 7 and 17 (k = 0), members B, L, C
           with the forced gate (lam, pi) = (-1, 0), so L stops at its first scored
           write and C blocks, blocks, then stops; block 2: tasks 8 and 1, members B
           and L with the tie rule at the cap, (15, 0.5). Seeds 890000 + u.
  pilot    all 50 tasks x 3 seed indices = 150 units, two pass-all members P and Q
           per unit (same task, same seed), 300 episodes; seed 810000 + u with
           u = 3 j + k; 3 blocks of 50 units (100 episodes), both members of a unit
           in one block; inert gate (15, 0) recorded on every block.
  confirm  n pairs (n a multiple of 50 chosen by rule G3-E3), balanced: every task
           gets m = n / 50 seed indices k = 0..m-1; u = 50 k + j, seed 830000 + u.
           Members B (pass-all) and L (live stopping gate (lam, pi)) for every unit;
           member C (live retry gate, arm C rule, cap 2) for the units with k < 8,
           i.e. the first 400 units, the same 8 seeds of every task. Block b holds
           the units with k = b - 1 (50 units, 150 episodes when k < 8, else 100).
           Within a block the units run in a random order fixed here
           (random.Random(920 + b)), and member order rotates with the unit's
           position (B L C, L C B, C B L).

  python e3_make_manifest.py --kind pilot --out e3/manifest_e3_pilot.json
  python e3_make_manifest.py --kind confirm --n-pairs 600 --lam 3.2 --pi 0 --out e3/manifest_e3.json
"""

from __future__ import annotations

import argparse
import json
import random

TAU2_COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"   # tag v1.0.1
N_TASKS = 50
C_UNITS_K = 8                                               # member C on seed indices 0..7
INERT = {"lam": 15.0, "pi": 0.0}


def task_id(j: int) -> str:
    return str(j)


def ep(eid, block, j, k, u, seed, member, gate) -> dict:
    return {"episode_id": eid, "block": block, "task_index": j, "task_id": task_id(j), "seed_index": k,
            "unit": u, "seed": seed, "member": member, "gate": gate}


def smoke() -> dict:
    eps, blocks = [], []
    forced = {"kind": "stop", "lam": -1.0, "pi": 0.0}
    retry_forced = {"kind": "retry", "lam": -1.0, "pi": 0.0, "retry_cap": 2}
    for u, j in enumerate([7, 17]):
        for m, g in (("B", {"kind": "pass_all"}), ("L", forced), ("C", retry_forced)):
            eps.append(ep(f"e3s-{u:02d}-{m}", 1, j, 0, u, 890000 + u, m, g))
    blocks.append({"block": 1, "label": "block-01-S", "host": "h1", "stage": "smoke", "lam": -1.0, "pi": 0.0})
    tie = {"kind": "stop", "lam": 15.0, "pi": 0.5}
    for i, j in enumerate([8, 1]):
        u = 2 + i
        for m, g in (("B", {"kind": "pass_all"}), ("L", tie)):
            eps.append(ep(f"e3s-{u:02d}-{m}", 2, j, 0, u, 890000 + u, m, g))
    blocks.append({"block": 2, "label": "block-02-S", "host": "h1", "stage": "smoke", "lam": 15.0, "pi": 0.5})
    return {"experiment": "E3", "kind": "smoke", "tau2_commit": TAU2_COMMIT, "domain": "airline",
            "task_split": "base", "block_order": blocks, "episodes": eps}


def pilot() -> dict:
    eps, blocks = [], []
    for b in range(3):
        blocks.append({"block": b + 1, "label": f"block-{b + 1:02d}-P", "host": "h1", "stage": "pilot", **INERT})
    for j in range(N_TASKS):
        for k in range(3):
            u = 3 * j + k
            b = u // 50 + 1
            for m in ("P", "Q"):
                eps.append(ep(f"e3p-{u:03d}-{m}", b, j, k, u, 810000 + u, m, {"kind": "pass_all"}))
    return {"experiment": "E3", "kind": "pilot", "tau2_commit": TAU2_COMMIT, "domain": "airline",
            "task_split": "base", "block_order": blocks, "episodes": eps}


def confirm(n_pairs: int, lam: float, pi: float, hosts: list[str]) -> dict:
    assert n_pairs % N_TASKS == 0 and n_pairs >= 400, n_pairs
    m = n_pairs // N_TASKS
    eps, blocks = [], []
    rot = (("B", "L", "C"), ("L", "C", "B"), ("C", "B", "L"))
    gates = {"B": {"kind": "pass_all"}, "L": {"kind": "stop", "lam": lam, "pi": pi},
             "C": {"kind": "retry", "lam": lam, "pi": pi, "retry_cap": 2}}
    for k in range(m):
        b = k + 1
        host = hosts[k % len(hosts)]
        blocks.append({"block": b, "label": f"block-{b:03d}-{host}", "host": host, "stage": "confirm",
                       "lam": lam, "pi": pi})
        order = list(range(N_TASKS))
        random.Random(920 + b).shuffle(order)
        for pos, j in enumerate(order):
            u = N_TASKS * k + j
            members = [x for x in rot[pos % 3] if x != "C" or k < C_UNITS_K]
            for mem in members:
                eps.append(ep(f"e3{mem.lower()}-{u:04d}", b, j, k, u, 830000 + u, mem, gates[mem]))
    return {"experiment": "E3", "kind": "confirm", "tau2_commit": TAU2_COMMIT, "domain": "airline",
            "task_split": "base", "n_pairs": n_pairs, "lam": lam, "pi": pi, "c_units": C_UNITS_K * N_TASKS,
            "block_order": blocks, "episodes": eps}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("smoke", "pilot", "confirm"), required=True)
    ap.add_argument("--n-pairs", type=int)
    ap.add_argument("--lam", type=float)
    ap.add_argument("--pi", type=float, default=0.0)
    ap.add_argument("--hosts", default="h1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.kind == "smoke":
        man = smoke()
    elif a.kind == "pilot":
        man = pilot()
    else:
        man = confirm(a.n_pairs, a.lam, a.pi, a.hosts.split(","))
    with open(a.out, "w") as f:
        json.dump(man, f, indent=1)
    print(a.out, len(man["episodes"]), "episodes", len(man["block_order"]), "blocks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
