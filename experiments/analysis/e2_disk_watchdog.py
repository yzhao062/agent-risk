"""E2 disk watchdog: the D66 watchdog (disk_watchdog_d66.py, imported unchanged) with an
E2 trip threshold, a size input that counts deleted-but-open files, and a record of
sandbox sizes.

The size input (since 2026-10-05, after the block-2 incident of the confirmatory run).
Each listed Inspect sandbox's size is its writable layer (`docker ps --size`, the D66
input) plus the allocated bytes of deleted-but-open files its processes hold
(disk_deleted_open.py, a root helper run through `sudo -n`). A file deleted while a
process keeps writing to it fills the disk but is invisible to `docker ps --size`; in
block 2 such a sandbox held about 20 GB while listing 26 MB, and the D66 input made the
watchdog stop 9 other sandboxes first. If the helper fails or times out, the watchdog
logs "helper_error" and ranks by the writable layers alone (the D66 input); a helper
error never stops the watchdog. The decision rule is unchanged.

Why a different threshold. On <HOST> Docker's data root is on the root file
system (60 GB, about 14.8 GB free before the pilot), and the weights volume stays
read-only (the coordinator's conditions). The D66 hosts put Docker on a large attached
volume. A runaway writer is stopped within the trip latency L (1 s polling, the
`docker ps --size` listing, `docker kill`); the root file system fills only if more
than the trip threshold T is written within L. Measured on this host (a buffered dd
from /dev/zero inside the sandbox image, the D65 pattern), free space on / falls by up to
1.9 GB in one second for one writer and 4.0 GB for two. E2 sets T = 20,000 MB (raised from
10,000 MB after the pilot, once the unused 14B weights left / and about 30 GB were free; the
coordinator's decision) and polls every POLL_S = 0.25 s (the D66 loop polls every 1 s), and
e2_run.sh starts a block only with at least 21,000 MB free (MIN_FREE_MB). The
pre-registration states the resulting margins.

The size record. Every SIZES_S seconds the watchdog lists the running Inspect
sandboxes' writable-layer sizes and logs the largest five and the total (event
"sizes"), so the pilot measures per-episode disk growth for the confirmatory plan.
The decision rule is the D66 one (Watchdog.step, unchanged): below T, stop the
largest Inspect sandbox with `docker kill`; the next stop only after a further
256 MB drop; free disk back at or above T ends the intervention. One watchdog
per host (the D66 lock file is shared, so a D66 watchdog and this one never run
together).

  python3 ~/harness/e2_disk_watchdog.py run --log ~/logs/e2/watchdog_h1.jsonl
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import disk_watchdog_d66 as d66  # noqa: E402  (unchanged D66 module)
import disk_deleted_open as ddo  # noqa: E402  (shared helper: deleted-but-open bytes per container)

E2_THRESHOLD_MB = 20000
POLL_S = 0.25
SIZES_S = 30.0


def run(log_path: str, threshold_mb: int = E2_THRESHOLD_MB) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    lock = open(d66.LOCK_PATH, "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        with open(log_path, "a") as f:
            t = time.time()
            f.write(json.dumps({"event": "already_running", "time": t, "utc": d66._utc(t),
                                "note": "another watchdog holds " + d66.LOCK_PATH + "; this one exits"}) + "\n")
        return 0
    paths = d66.data_paths()

    def log(rec: dict) -> None:
        if rec.get("event") == "heartbeat":
            rec["root_fs_free_mb"] = d66.free_mb("/")
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    def read_sizes() -> list[tuple[str, int]]:
        return ddo.sizes_with_deleted(d66.docker_sizes(), log)
    wd = d66.Watchdog(lambda: d66.min_free_mb(paths), read_sizes, d66.docker_kill, log,
                      threshold_mb=threshold_mb)
    t0 = time.time()
    log({"event": "start", "time": t0, "utc": d66._utc(t0), "threshold_mb": wd.threshold_mb,
         "redrop_mb": wd.redrop_mb, "poll_s": POLL_S, "paths": {p: d66.free_mb(p) for p in paths},
         "root_fs_free_mb": d66.free_mb("/"), "pid": os.getpid(),
         "watchdog": "e2 (D66 rule, E2 threshold, sizes include deleted-but-open files)"})
    nxt = time.monotonic()
    last_sizes = -1e18
    while True:
        try:
            wd.step()
            now = time.time()
            if now - last_sizes >= SIZES_S and not wd.in_intervention:
                t1 = time.monotonic()
                base = d66.docker_sizes()
                sz = sorted(ddo.sizes_with_deleted(base, log), key=lambda x: -x[1])
                log({"event": "sizes", "time": now, "utc": d66._utc(now), "n": len(sz),
                     "total_bytes": sum(s for _, s in sz), "layer_bytes": sum(s for _, s in base),
                     "largest": [[n, s] for n, s in sz[:5]],
                     "list_seconds": round(time.monotonic() - t1, 3), "free_mb": d66.min_free_mb(paths)})
                last_sizes = now
        except Exception as exc:  # noqa: BLE001
            log({"event": "error", "time": time.time(), "error": str(exc)[:200]})
        nxt += POLL_S
        time.sleep(max(0.0, nxt - time.monotonic()))


def all_sizes(timeout: int = 20) -> list[tuple[str, int]]:
    """Writable-layer sizes of every running container (dry run only; the watchdog itself
    ranks Inspect sandboxes only)."""
    import subprocess
    r = subprocess.run(["docker", "ps", "--size", "--format", "{{.Names}}\t{{.Size}}"],
                       capture_output=True, text=True, timeout=timeout)
    return [(n, d66.parse_size(sz)) for n, _, sz in (l.partition("\t") for l in (r.stdout or "").splitlines())]


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run")
    a.add_argument("--log", required=True)
    a.add_argument("--threshold-mb", type=int, default=E2_THRESHOLD_MB)
    b = sub.add_parser("sizes", help="dry run: print the size input the watchdog would rank (changes nothing)")
    b.add_argument("--all", action="store_true", help="every running container, not only Inspect sandboxes")
    args = ap.parse_args()
    if args.cmd == "sizes":
        errs: list = []
        base = all_sizes() if args.all else d66.docker_sizes()
        merged = ddo.sizes_with_deleted(base, errs.append)
        print(json.dumps({"layer": dict(base), "with_deleted": dict(merged), "helper_errors": errs}))
        return 0
    return run(os.path.expanduser(args.log), args.threshold_mb)


if __name__ == "__main__":
    raise SystemExit(main())
