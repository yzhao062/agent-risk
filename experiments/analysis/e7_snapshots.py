"""E7 snapshot images: deletion after use, inventory (VM side; D66's snapshot_tools does the Docker calls).

A base episode leaves one image per step above the floor (arh-e7:<unit>-b<step>). After block k has
a finished attempt, every image of the units of chunk k - 1 (the units whose cont episodes block k
planned) is deleted (docker image rm per tag, then docker image prune -f), whether or not each
continuation ran. Snapshots are not archived; the episode logs are.

  PYTHONPATH=~/harness python ~/harness/e7_snapshots.py cleanup-block --manifest e7/manifest_e7.json --block 7
  PYTHONPATH=~/harness python ~/harness/e7_snapshots.py inventory --manifest e7/manifest_e7.json --host 1
  PYTHONPATH=~/harness python ~/harness/e7_snapshots.py prune-host --manifest e7/manifest_e7.json --host 1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

from agent_risk_harness.e7 import SNAPSHOT_REPO
from snapshot_tools import remove_tags


def local_tags() -> list[str]:
    r = subprocess.run(["docker", "images", SNAPSHOT_REPO, "--format", "{{.Repository}}:{{.Tag}}"],
                       capture_output=True, text=True, timeout=300)
    return [x.strip() for x in (r.stdout or "").splitlines() if x.strip() and "<none>" not in x]


def unit_tags(unit_ids: list[str], tags: list[str] | None = None) -> list[str]:
    """Local snapshot tags of the given base unit ids: arh-e7:<unit>-b<step>."""
    tags = local_tags() if tags is None else tags
    prefixes = tuple(f"{SNAPSHOT_REPO}:{u}-b".lower() for u in unit_ids)
    return [t for t in tags if t.startswith(prefixes)]


def block_unit_ids(manifest: dict, block: int) -> list[str]:
    """Base unit ids whose cont episodes the block plans (chunk k - 1)."""
    return sorted({e["base_unit_id"] for e in manifest["episodes"]
                   if int(e["block"]) == int(block) and e["condition"] == "cont"})


def host_unit_ids(manifest: dict, host: int) -> list[str]:
    return [u["unit_id"] for u in manifest["units"] if int(u["host"]) == int(host)]


def inventory(tags: list[str]) -> dict:
    size = 0
    for t in tags:
        r = subprocess.run(["docker", "image", "inspect", t, "--format", "{{.Size}}"], capture_output=True, text=True,
                           timeout=120)
        if r.returncode == 0 and r.stdout.strip().isdigit():
            size += int(r.stdout.strip())
    return {"present": len(tags), "present_tags": tags[:100], "total_size_bytes_incl_shared_base": size}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cleanup-block", "inventory", "prune-host"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--block", type=int)
    ap.add_argument("--host", type=int)
    args = ap.parse_args()
    man = json.load(open(os.path.expanduser(args.manifest)))
    if args.cmd == "inventory":
        print(json.dumps(inventory(unit_tags(host_unit_ids(man, args.host)))))
        return 0
    ids = block_unit_ids(man, args.block) if args.cmd == "cleanup-block" else host_unit_ids(man, args.host)
    out = remove_tags(unit_tags(ids))
    if not _clean(out):
        time.sleep(5)                       # one retry on what is left, the failure report is kept
        first = out
        out = remove_tags(unit_tags(ids))
        out["first_attempt"] = first
    print(json.dumps(out))
    return 0 if _clean(out) else 4


def _clean(out: dict) -> bool:
    """A removal report without failed tags and with a successful prune (exit status 4 otherwise, so the chain
    does not mark the block's cleanup as done)."""
    return not out.get("failed") and out.get("prune_rc") in (0, None)


if __name__ == "__main__":
    raise SystemExit(main())
