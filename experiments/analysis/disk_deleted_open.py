"""Disk held by deleted-but-open files, per Docker container (shared by every harness's
disk watchdog; first used by e2_disk_watchdog.py).

Why: `docker ps --size` reports a sandbox's writable layer from the files that still
have a path. A process that keeps writing to a file that has been deleted while it is
open (for example `cat f >> f` after `rm f`) fills the disk while its sandbox still
shows a small layer. In the E2 confirmatory run (block 2, 2026-10-05T05:10Z) one
sandbox held about 20 GB this way while listing 26 MB, so the D66 watchdog, which
stops the largest listed layer, stopped 9 other sandboxes first.

Two parts:

  scan()               root side: walks /proc/<pid>/fd, keeps regular files whose
                       link target ends in " (deleted)", counts each open file once
                       (device and inode) with its allocated bytes (st_blocks * 512),
                       and assigns it to the 64-hex Docker container id found in
                       /proc/<pid>/cgroup. Reads /proc only; changes nothing.
                         sudo -n python3 ~/harness/disk_deleted_open.py
                       prints {"<container id>": bytes, ...}.
  sizes_with_deleted() watchdog side: adds each listed container's deleted-but-open
                       bytes to its writable-layer size. If the root helper fails,
                       times out or returns something unreadable, it logs a
                       "helper_error" record and returns the listed sizes unchanged,
                       so the watchdog falls back to the D66 size input and never
                       stops working because of the helper.

The watchdog's decision rule (disk_watchdog_d66.Watchdog.step) is not touched; only
the sizes it ranks change.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
import time

CID_RE = re.compile(r"[0-9a-f]{64}")
HELPER_TIMEOUT_S = 10


def container_of(cgroup_text: str) -> str | None:
    m = CID_RE.search(cgroup_text or "")
    return m.group(0) if m else None


def scan(proc: str = "/proc", listdir=os.listdir, readlink=os.readlink, fstat=os.stat,
         read=lambda p: open(p).read()) -> dict[str, int]:
    """{container id: allocated bytes of deleted-but-open regular files}; I/O injected for tests."""
    seen: set[tuple[int, int]] = set()
    out: dict[str, int] = {}
    for pid in listdir(proc):
        if not pid.isdigit():
            continue
        try:
            cid = container_of(read(os.path.join(proc, pid, "cgroup")))
        except OSError:
            continue
        if cid is None:
            continue
        fddir = os.path.join(proc, pid, "fd")
        try:
            fds = listdir(fddir)
        except OSError:
            continue
        for fd in fds:
            p = os.path.join(fddir, fd)
            try:
                if not readlink(p).endswith(" (deleted)"):
                    continue
                st = fstat(p)
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            key = (st.st_dev, st.st_ino)
            if key in seen:
                continue
            seen.add(key)
            out[cid] = out.get(cid, 0) + int(getattr(st, "st_blocks", 0)) * 512
    return out


def run_helper(timeout: float = HELPER_TIMEOUT_S) -> dict[str, int]:
    """Run scan() as root through `sudo -n`; raises on any failure."""
    r = subprocess.run(["sudo", "-n", sys.executable or "python3", os.path.abspath(__file__)],
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"helper exit {r.returncode}: {(r.stderr or '')[:200]}")
    data = json.loads(r.stdout)
    if not isinstance(data, dict):
        raise ValueError("helper output is not an object")
    return {str(k): int(v) for k, v in data.items()}


def container_names(timeout: float = 20) -> dict[str, str]:
    """{full container id: name} for running containers."""
    r = subprocess.run(["docker", "ps", "--no-trunc", "--format", "{{.ID}}\t{{.Names}}"],
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"docker ps exit {r.returncode}")
    out = {}
    for line in (r.stdout or "").splitlines():
        cid, _, name = line.partition("\t")
        if cid:
            out[cid.strip()] = name.strip()
    return out


def sizes_with_deleted(base: list[tuple[str, int]], log=None, helper=run_helper, names=container_names,
                       clock=time.time) -> list[tuple[str, int]]:
    """Listed (name, layer bytes) plus each container's deleted-but-open bytes.
    Any failure of the helper or the name lookup: log "helper_error" and return base."""
    try:
        by_id = helper()
        id_name = names() if by_id else {}
        by_name: dict[str, int] = {}
        for cid, b in by_id.items():
            n = id_name.get(cid)
            if n is not None:
                by_name[n] = by_name.get(n, 0) + int(b)
        return [(n, int(s) + by_name.get(n, 0)) for n, s in base]
    except Exception as exc:  # noqa: BLE001  (fallback: the D66 size input)
        if log is not None:
            t = clock()
            log({"event": "helper_error", "time": t, "error": f"{type(exc).__name__}: {exc}"[:300],
                 "fallback": "writable-layer sizes only (D66 size input)"})
        return list(base)


def main() -> int:
    print(json.dumps(scan()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
