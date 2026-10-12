"""Host-side Docker helpers for the D66 relay snapshots.

The handoff solver (agent_risk_harness/relay.py) commits each ``relay`` and
``first`` container to ``arh-relay:<unit_id>-<mode>``, with the handoff digest
as the label ``arh.relay.digest``. The snapshot-start modes read those images
through one compose file per unit: the pilot compose (same CPU, memory, init,
command and DNS) with the committed image in place of the build.

After a block's attempt finishes, run_relay.py deletes the snapshots that the
block's B part consumed (remove_tags of block_source_tags), so at most four
chunks of snapshots (64) exist at once. Snapshots are not archived.

CLI (on the VM, from ~/harness; PYTHONPATH=~/harness):
  python snapshot_tools.py inventory-half --manifest d66/manifest_d66.json --half a
  python snapshot_tools.py cleanup-block  --manifest d66/manifest_d66.json --block 7
  python snapshot_tools.py prune-half     --manifest d66/manifest_d66.json --half a
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess

from agent_risk_harness.relay import FROM_SNAPSHOT, SNAPSHOT_MODES, snapshot_tag

COMPOSE_TEMPLATE = """# D66 snapshot start: the pilot sandbox (agent_risk_harness/sandbox/compose.yml)
# with the committed handoff image {image} in place of the build.
services:
  default:
    image: {image}
    x-local: true
    init: true
    command: tail -f /dev/null
    cpus: 1.0
    mem_limit: 0.5gb
    dns:
      - 192.0.2.1
"""


def compose_text(image: str) -> str:
    return COMPOSE_TEMPLATE.format(image=image)


def write_compose(directory: str, unit_id: str, source_mode: str) -> str:
    os.makedirs(directory, exist_ok=True)
    tag = snapshot_tag(unit_id, source_mode)
    path = os.path.abspath(os.path.join(directory, f"compose-{unit_id}-{source_mode}.yml"))
    with open(path, "w") as f:
        f.write(compose_text(tag))
    return path


def _docker(args: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


def image_info(tag: str) -> dict | None:
    """{'id', 'size', 'digest_label'} of a local image, or None if absent."""
    r = _docker(["image", "inspect", tag, "--format", "{{json .}}"])
    if r.returncode != 0:
        return None
    try:
        d = json.loads(r.stdout.strip())
    except json.JSONDecodeError:
        return None
    labels = ((d.get("Config") or {}).get("Labels")) or {}
    return {"id": d.get("Id"), "size": d.get("Size"), "digest_label": labels.get("arh.relay.digest"),
            "unit_label": labels.get("arh.relay.unit")}


def remove_tags(tags: list[str]) -> dict:
    """docker image rm for each present tag, then docker image prune -f (dangling layers)."""
    removed, failed, absent = [], [], []
    for t in tags:
        if image_info(t) is None:
            absent.append(t)
            continue
        r = _docker(["image", "rm", t], timeout=600)
        (removed if r.returncode == 0 else failed).append(t)
    pr = _docker(["image", "prune", "-f"], timeout=600)
    return {"removed": len(removed), "absent": len(absent), "failed": failed,
            "prune_rc": pr.returncode, "prune_out": (pr.stdout or "").strip().splitlines()[-1:]}


def block_source_tags(manifest: dict, block: int) -> list[str]:
    """Snapshot tags consumed by a block's B part (second_snap: first; second_rep: relay)."""
    return [snapshot_tag(e["unit_id"], FROM_SNAPSHOT[e["mode"]]) for e in manifest["episodes"]
            if int(e["block"]) == int(block) and e["mode"] in FROM_SNAPSHOT]


def half_tags(manifest: dict, half: str) -> list[str]:
    return [snapshot_tag(u["unit_id"], m) for u in manifest["units"] if u["half"] == half for m in SNAPSHOT_MODES]


def inventory(tags: list[str]) -> dict:
    present, size = [], 0
    for t in tags:
        info = image_info(t)
        if info is not None:
            present.append(t)
            size += int(info.get("size") or 0)
    return {"present": len(present), "present_tags": present[:100], "total_size_bytes": size}


def prune(manifest: dict) -> dict:
    """Remove every snapshot of the manifest's units (scripted smoke cleanup)."""
    return remove_tags([snapshot_tag(u["unit_id"], m) for u in manifest["units"] for m in SNAPSHOT_MODES])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["inventory-half", "cleanup-block", "prune-half"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--block", type=int)
    ap.add_argument("--half")
    args = ap.parse_args()
    man = json.load(open(os.path.expanduser(args.manifest)))
    if args.cmd == "inventory-half":
        out = inventory(half_tags(man, args.half))
    elif args.cmd == "cleanup-block":
        out = remove_tags(block_source_tags(man, args.block))
    else:
        out = remove_tags(half_tags(man, args.half))
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FROM_SNAPSHOT", "compose_text", "write_compose", "image_info", "remove_tags", "block_source_tags",
           "half_tags", "inventory", "prune"]
