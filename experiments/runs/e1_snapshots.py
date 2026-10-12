"""E1 snapshot images: deletion after use, inventory (VM side; D66's snapshot_tools
does the Docker calls).

After block k has a finished attempt, the snapshots of the units whose cont
episodes block k planned (chunk k - 1) are deleted (docker image rm, then
docker image prune -f), whether or not each continuation ran. Snapshots are not
archived; the episode logs are.

  PYTHONPATH=~/harness python ~/harness/e1_snapshots.py cleanup-block --manifest e1/manifest_e1.json --block 7
  PYTHONPATH=~/harness python ~/harness/e1_snapshots.py inventory-half --manifest e1/manifest_e1.json --half a
  PYTHONPATH=~/harness python ~/harness/e1_snapshots.py prune-half --manifest e1/manifest_e1.json --half a
"""

from __future__ import annotations

import argparse
import json
import os

from agent_risk_harness.e1 import snapshot_tag
from snapshot_tools import inventory, remove_tags


def block_tags(manifest: dict, block: int) -> list[str]:
    return [snapshot_tag(e["unit_id"]) for e in manifest["episodes"]
            if int(e["block"]) == int(block) and e["condition"] == "cont"]


def half_tags(manifest: dict, half: str) -> list[str]:
    return [snapshot_tag(u["unit_id"]) for u in manifest["units"] if u["half"] == half]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cleanup-block", "inventory-half", "prune-half"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--block", type=int)
    ap.add_argument("--half")
    args = ap.parse_args()
    man = json.load(open(os.path.expanduser(args.manifest)))
    if args.cmd == "cleanup-block":
        out = remove_tags(block_tags(man, args.block))
    elif args.cmd == "inventory-half":
        out = inventory(half_tags(man, args.half))
    else:
        out = remove_tags(half_tags(man, args.half))
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
