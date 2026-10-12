"""D66 disk watchdog (pre-launch revision of D66-PREREG-2026-10-04.md).

Why: in D65 on VM2 one sandbox wrote about 15 GB in about 20 s; free disk fell
from 14,462 MB to 25 MB before the 10 s, 2 GB disk_watchdog.sh stopped it, and
episodes in flight failed with "No space left on device". D66 also keeps up to
64 snapshots on the same disk. disk_watchdog.sh is left as it is; D66 runs this
one instead.

Cause of the D65 burst (coordinator's record): episode d65-0225's task asked the
agent to simulate heavy disk I/O, and it ran
`dd if=/dev/zero of=/tmp/simulated_io bs=1M count=1000000 &` (about 1 TB) at
about 750 MB/s. The same task can be drawn in D66; no task is excluded (that
would change the registered task distribution). Disk headroom and this
watchdog are the protection, and every intervention is recorded per episode.

Where free space is read: the file systems that hold the sandboxes' writable
layers and the snapshots, i.e. Docker's data root (`docker info`
DockerRootDir) and containerd's root (/var/lib/containerd, which holds the
layers when Docker uses the containerd image store), whichever exist; free
space is the smallest over them. The run VMs move Docker's data root to a
large attached volume, so this is not the root file system. The root file
system's free space (the logs) is logged with each heartbeat.

Rule (run):
  - every POLL_S = 1 s, read the free space of the data file systems (statvfs,
    space available to non-root, the smallest over them);
  - while free >= THRESHOLD_MB (6,000 MB), do nothing (a heartbeat line every
    HEARTBEAT_S);
  - when free < THRESHOLD_MB: list running Inspect sandboxes with their
    writable-layer size (`docker ps --size`), stop the largest with
    `docker kill`, and log the intervention (time, free disk, target, size, the
    five largest layers);
  - after a kill, the killed container's layer stays on disk until Inspect
    removes the container, so free disk does not recover at once; the next kill
    happens only if free disk falls a further REDROP_MB (256 MB) below its level
    at the last kill (another writer). Free disk back at or above the threshold
    ends the intervention.

Threshold: a burst at the D65 rate (about 750 MB/s) crossing 6,000 MB is
stopped within about 1 s of polling plus the `docker ps --size` call and the
kill (a few seconds), which leaves about 2 GB; a block's own commits need at
most a few hundred MB at the measured 50 MB per snapshot. The threshold is an
absolute amount, so it does not depend on the size of the data volume.
run_relay.sh starts a block only with at least 8,000 MB free on the same data
file systems, above this threshold. A per-container writable-layer cap
(`--storage-opt size=...`) works only with storage drivers that support it
(overlay2 on xfs mounted with pquota, among others); the run agent checks it on
the attached volume, and this watchdog stays the backstop either way.

episodes: lists every planned episode whose run window (first started.jsonl
line to its "end" line; an episode without an end line runs to its attempt's
finish, or is open-ended) contains an intervention time on its own host (the
log watchdog_<half>.jsonl of the episode's half), and the episodes whose
recorded container (RelayState.container) was the one stopped. One D66
watchdog runs per host (a lock file; a second one logs "already_running" and
exits). These
episodes stay in every denominator under the registered missing-outcome rules;
the list is descriptive.

  python3 ~/harness/disk_watchdog_d66.py run --log ~/logs/d66/watchdog_a.jsonl
  python3 ~/harness/disk_watchdog_d66.py free [--json]
  python3 disk_watchdog_d66.py episodes --manifest d66/manifest_d66.json --logroot logs/d66 \
      --out logs/d66/watchdog_episodes.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

THRESHOLD_MB = 6000
REDROP_MB = 256
POLL_S = 1.0
HEARTBEAT_S = 30.0
NAME_PREFIX = "inspect-"          # Inspect's compose projects; nothing else is ever stopped
LOCK_PATH = "/tmp/arh_disk_watchdog_d66.lock"   # one D66 watchdog per host

_UNITS = {"B": 1, "kB": 1e3, "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12,
          "KiB": 1024, "MiB": 1024 ** 2, "GiB": 1024 ** 3, "TiB": 1024 ** 4}


def parse_size(text: str) -> int:
    """Writable-layer bytes from docker's Size column, e.g. '15.2GB (virtual 16.1GB)'."""
    m = re.match(r"\s*([0-9.]+)\s*([A-Za-z]+)", text or "")
    if not m or m.group(2) not in _UNITS:
        return 0
    return int(float(m.group(1)) * _UNITS[m.group(2)])


def free_mb(path: str = "/") -> int:
    st = os.statvfs(path)
    return int(st.f_bavail * st.f_frsize // (1024 * 1024))


def data_paths() -> list[str]:
    """Docker's data root and containerd's root, one path per file system; "/" if neither is found."""
    paths = []
    try:
        r = subprocess.run(["docker", "info", "--format", "{{.DockerRootDir}}"], capture_output=True, text=True,
                           timeout=30)
        d = (r.stdout or "").strip()
        if r.returncode == 0 and d and os.path.isdir(d):
            paths.append(d)
    except Exception:  # noqa: BLE001
        pass
    if os.path.isdir("/var/lib/containerd"):
        paths.append("/var/lib/containerd")
    out, devs = [], set()
    for p in paths or ["/"]:
        try:
            dev = os.stat(p).st_dev
        except OSError:
            continue
        if dev not in devs:
            devs.add(dev)
            out.append(p)
    return out or ["/"]


def min_free_mb(paths: list[str]) -> int:
    return min(free_mb(p) for p in paths)


def docker_sizes(timeout: int = 20) -> list[tuple[str, int]]:
    r = subprocess.run(["docker", "ps", "--size", "--format", "{{.Names}}\t{{.Size}}"],
                       capture_output=True, text=True, timeout=timeout)
    out = []
    for line in (r.stdout or "").splitlines():
        name, _, size = line.partition("\t")
        if name.startswith(NAME_PREFIX):
            out.append((name, parse_size(size)))
    return out


def docker_kill(name: str) -> bool:
    r = subprocess.run(["docker", "kill", name], capture_output=True, text=True, timeout=30)
    return r.returncode == 0


def _utc(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat()


class Watchdog:
    """One decision per poll; the readers, the kill and the clock are injected (tests)."""

    def __init__(self, read_free, read_sizes, kill, log, clock=time.time,
                 threshold_mb: int = THRESHOLD_MB, redrop_mb: int = REDROP_MB, heartbeat_s: float = HEARTBEAT_S):
        self.read_free, self.read_sizes, self.kill, self.log, self.clock = read_free, read_sizes, kill, log, clock
        self.threshold_mb, self.redrop_mb, self.heartbeat_s = threshold_mb, redrop_mb, heartbeat_s
        self.last_kill_free: int | None = None
        self.killed: set[str] = set()
        self.last_beat = -1e18
        self.in_intervention = False

    def step(self) -> dict | None:
        now = self.clock()
        free = self.read_free()
        if free >= self.threshold_mb:
            if self.in_intervention:
                self.log({"event": "recovered", "time": now, "utc": _utc(now), "free_mb": free})
            self.in_intervention = False
            self.last_kill_free = None
            if now - self.last_beat >= self.heartbeat_s:
                self.log({"event": "heartbeat", "time": now, "utc": _utc(now), "free_mb": free})
                self.last_beat = now
            return None
        self.in_intervention = True
        if self.last_kill_free is not None and free > self.last_kill_free - self.redrop_mb:
            return None                     # waiting for the stopped container's removal
        try:
            sizes = [(n, s) for n, s in self.read_sizes() if n not in self.killed]
        except Exception as exc:  # noqa: BLE001
            self.log({"event": "error", "time": now, "utc": _utc(now), "free_mb": free, "error": str(exc)[:200]})
            return None
        if not sizes:
            self.log({"event": "no_target", "time": now, "utc": _utc(now), "free_mb": free})
            self.last_kill_free = free
            return None
        sizes.sort(key=lambda x: -x[1])
        target, size = sizes[0]
        ok = self.kill(target)
        self.killed.add(target)
        after = self.read_free()
        # the hold level is the free space once the target has stopped (the size listing takes time)
        self.last_kill_free = min(free, after)
        rec = {"event": "intervention", "time": now, "utc": _utc(now), "free_mb": free, "target": target,
               "size_bytes": size, "kill_ok": ok, "n_candidates": len(sizes),
               "largest": [[n, s] for n, s in sizes[:5]], "free_mb_after_kill": after}
        self.log(rec)
        return rec


def run(log_path: str, paths: list[str] | None = None, lock_path: str = LOCK_PATH) -> int:
    import fcntl
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    lock = open(lock_path, "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        with open(log_path, "a") as f:
            t = time.time()
            f.write(json.dumps({"event": "already_running", "time": t, "utc": _utc(t),
                                "note": "another D66 watchdog holds " + lock_path + "; this one exits"}) + "\n")
        return 0
    paths = paths or data_paths()

    def log(rec: dict) -> None:
        if rec.get("event") == "heartbeat":
            rec["root_fs_free_mb"] = free_mb("/")
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    wd = Watchdog(lambda: min_free_mb(paths), docker_sizes, docker_kill, log)
    t0 = time.time()
    log({"event": "start", "time": t0, "utc": _utc(t0), "threshold_mb": wd.threshold_mb, "redrop_mb": wd.redrop_mb,
         "poll_s": POLL_S, "paths": {p: free_mb(p) for p in paths}, "root_fs_free_mb": free_mb("/"),
         "pid": os.getpid()})
    nxt = time.monotonic()
    while True:
        try:
            wd.step()
        except Exception as exc:  # noqa: BLE001
            log({"event": "error", "time": time.time(), "error": str(exc)[:200]})
        nxt += POLL_S
        time.sleep(max(0.0, nxt - time.monotonic()))


def interventions(logroot: str) -> list[dict]:
    """Interventions from watchdog_<half>.jsonl files; each applies to its half's host only."""
    out = []
    for p in sorted(glob.glob(os.path.join(logroot, "watchdog_*.jsonl"))):
        half = os.path.basename(p)[len("watchdog_"):-len(".jsonl")]
        with open(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("event") == "intervention":
                    out.append(dict(r, log=os.path.basename(p), half=half))
    return out


def episodes(manifest: dict, logroot: str) -> dict:
    """Episodes that ran during an intervention, and episodes whose container was stopped."""
    ivs = interventions(logroot)
    sid_to_eid = {f"{e['episode_id']}-t{e['task_id']}": e["episode_id"] for e in manifest["episodes"]}
    label_of = {e["episode_id"]: e["label"] for e in manifest["episodes"]}
    half_of = {e["episode_id"]: e["half"] for e in manifest["episodes"]}
    windows: dict[str, list[float]] = {}
    for blk in manifest["block_order"]:
        for adir in sorted(glob.glob(os.path.join(logroot, blk["label"], "attempt-*"))):
            fin = None
            rj = os.path.join(adir, "run.json")
            if os.path.exists(rj):
                try:
                    fin = datetime.fromisoformat(json.load(open(rj))["finished_utc"]).timestamp()
                except Exception:  # noqa: BLE001
                    fin = None
            sp = os.path.join(adir, "started.jsonl")
            if not os.path.exists(sp):
                continue
            span: dict[str, list] = {}
            with open(sp) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    eid = sid_to_eid.get(str(r.get("sample_id")))
                    if eid is None:
                        continue
                    w = span.setdefault(eid, [r["time"], None])
                    w[0] = min(w[0], r["time"])
                    if r.get("event") == "end":
                        w[1] = r["time"]
            for eid, (a, b) in span.items():
                windows[eid] = [a, b if b is not None else (fin if fin is not None else float("inf"))]
    during = sorted(e for e, (a, b) in windows.items()
                    if any(iv["half"] == half_of.get(e) and a <= iv["time"] <= b for iv in ivs))
    killed_names = {iv["target"] for iv in ivs}
    killed = []
    for p in glob.glob(os.path.join(logroot, "*", "attempt-*", "episodes.jsonl")):
        with open(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ((r.get("relay") or {}).get("container")) in killed_names:
                    killed.append(r.get("episode_id"))
    return {"n_interventions": len(ivs), "interventions": ivs, "n_episodes_with_window": len(windows),
            "ran_during_intervention": during, "by_block": {lb: sum(1 for e in during if label_of.get(e) == lb)
                                                            for lb in sorted({label_of.get(e) for e in during})},
            "container_stopped": sorted(set(killed)),
            "rule": "descriptive only; every listed episode stays in every denominator under the registered "
                    "missing-outcome rules"}


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run")
    a.add_argument("--log", required=True)
    a.add_argument("--path", action="append", default=None, help="override the data paths (repeatable)")
    c = sub.add_parser("free", help="print the smallest free MB over the data paths (run_relay.sh's guard)")
    c.add_argument("--json", action="store_true")
    b = sub.add_parser("episodes")
    b.add_argument("--manifest", required=True)
    b.add_argument("--logroot", required=True)
    b.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "run":
        return run(os.path.expanduser(args.log), args.path)
    if args.cmd == "free":
        paths = data_paths()
        if args.json:
            print(json.dumps({"paths": {p: free_mb(p) for p in paths}, "min_free_mb": min_free_mb(paths),
                              "root_fs_free_mb": free_mb("/")}))
        else:
            print(min_free_mb(paths))
        return 0
    res = episodes(json.load(open(args.manifest)), args.logroot)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("n_interventions", "n_episodes_with_window")}
                     | {"n_ran_during_intervention": len(res["ran_during_intervention"]),
                        "n_container_stopped": len(res["container_stopped"])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
