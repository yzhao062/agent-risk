"""E1 disk watchdog: the D66 watchdog (disk_watchdog_d66.py, imported unchanged) whose
size input also counts deleted-but-open files (disk_deleted_open.py, the helper shared
with E2). Safety-only change during the E1 confirmatory run (E1-PREREG-2026-10-05.md,
Revision Record, "Safety-only change during the run").

Why. In E2's confirmatory run a sandbox appended a file to itself after deleting it
(about 20 GB held by a deleted-but-open file). `docker ps --size` lists only files that
still have a path, so the D66 size input showed that sandbox at 26 MB, and the D66 rule
stopped 9 sandboxes that were not writing before a person stopped the writer. E1 runs
the same task distribution, so the same blind spot applies to its batch watchdogs.

What changes. Only the sizes the D66 rule ranks: each listed Inspect sandbox's size is
its writable layer (`docker ps --size`, the D66 input) plus the allocated bytes of the
deleted-but-open files its processes hold (`sudo -n python3 disk_deleted_open.py`, which
reads /proc and changes nothing). If the helper fails, times out or returns something
unreadable, the watchdog logs "helper_error" and ranks by the writable layers alone,
which is the D66 input; a helper error never stops the watchdog.

What does not change. The decision rule (disk_watchdog_d66.Watchdog.step), the E1
threshold (6,000 MB free on the data file systems), the 256 MB re-drop, the 1 s polling,
the 30 s heartbeat, the data paths, the D66 lock (one watchdog per host, shared with
disk_watchdog_d66.py), the log path and its event names (start, heartbeat,
intervention, recovered, no_target, error, already_running), so
`disk_watchdog_d66.py episodes` reads the same log unchanged. New events: one
"helper_check" after start (the helper run once: seconds, containers with deleted
bytes, total bytes, or the error) and "helper_error" when a ranking falls back.

  python3 ~/harness/e1_disk_watchdog.py run --log ~/logs/e1/watchdog_a.jsonl
  python3 ~/harness/e1_disk_watchdog.py sizes [--all]     # read-only dry run
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import disk_watchdog_d66 as d66  # noqa: E402  (unchanged D66 module: rule, threshold, lock)
import disk_deleted_open as ddo  # noqa: E402  (shared helper: deleted-but-open bytes per container)

POLL_S = d66.POLL_S                 # 1 s, as registered for E1


def make_read_sizes(log, docker_sizes=d66.docker_sizes, helper=ddo.run_helper, names=ddo.container_names):
    """The size input the D66 rule ranks: listed layers plus deleted-but-open bytes (fallback: layers)."""
    def read_sizes() -> list[tuple[str, int]]:
        return ddo.sizes_with_deleted(docker_sizes(), log, helper=helper, names=names)
    return read_sizes


def make_watchdog(read_free, kill, log, docker_sizes=d66.docker_sizes, helper=ddo.run_helper,
                  names=ddo.container_names, clock=time.time) -> d66.Watchdog:
    """D66 Watchdog with its default threshold, re-drop and heartbeat; only read_sizes differs."""
    return d66.Watchdog(read_free, make_read_sizes(log, docker_sizes, helper, names), kill, log, clock=clock)


def helper_check(helper=ddo.run_helper) -> dict:
    t = time.time()
    t1 = time.monotonic()
    try:
        got = helper()
        return {"event": "helper_check", "time": t, "utc": d66._utc(t), "ok": True,
                "seconds": round(time.monotonic() - t1, 3), "n_containers_with_deleted": len(got),
                "deleted_bytes_total": int(sum(got.values()))}
    except Exception as exc:  # noqa: BLE001
        return {"event": "helper_check", "time": t, "utc": d66._utc(t), "ok": False,
                "seconds": round(time.monotonic() - t1, 3), "error": f"{type(exc).__name__}: {exc}"[:300],
                "fallback": "writable-layer sizes only (D66 size input) whenever the helper fails"}


def run(log_path: str) -> int:
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
    wd = make_watchdog(lambda: d66.min_free_mb(paths), d66.docker_kill, log)
    t0 = time.time()
    log({"event": "start", "time": t0, "utc": d66._utc(t0), "threshold_mb": wd.threshold_mb,
         "redrop_mb": wd.redrop_mb, "poll_s": POLL_S, "paths": {p: d66.free_mb(p) for p in paths},
         "root_fs_free_mb": d66.free_mb("/"), "pid": os.getpid(),
         "watchdog": "e1 (D66 rule and threshold, sizes include deleted-but-open files)"})
    log(helper_check())
    nxt = time.monotonic()
    while True:
        try:
            wd.step()
        except Exception as exc:  # noqa: BLE001
            log({"event": "error", "time": time.time(), "error": str(exc)[:200]})
        nxt += POLL_S
        time.sleep(max(0.0, nxt - time.monotonic()))


def all_sizes(timeout: int = 20) -> list[tuple[str, int]]:
    """Writable-layer sizes of every running container (dry run only; the watchdog ranks Inspect sandboxes only)."""
    r = subprocess.run(["docker", "ps", "--size", "--format", "{{.Names}}\t{{.Size}}"],
                       capture_output=True, text=True, timeout=timeout)
    return [(n, d66.parse_size(sz)) for n, _, sz in (ln.partition("\t") for ln in (r.stdout or "").splitlines())]


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run")
    a.add_argument("--log", required=True)
    b = sub.add_parser("sizes", help="dry run: print the size input the watchdog would rank (changes nothing)")
    b.add_argument("--all", action="store_true", help="every running container, not only Inspect sandboxes")
    args = ap.parse_args()
    if args.cmd == "sizes":
        errs: list = []
        t1 = time.monotonic()
        base = all_sizes() if args.all else d66.docker_sizes()
        merged = ddo.sizes_with_deleted(base, errs.append)
        print(json.dumps({"layer": dict(base), "with_deleted": dict(merged), "helper_errors": errs,
                          "seconds": round(time.monotonic() - t1, 3)}))
        return 0
    return run(os.path.expanduser(args.log))


if __name__ == "__main__":
    raise SystemExit(main())
