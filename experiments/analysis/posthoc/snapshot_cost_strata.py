"""Per-step cost strata of the 485 E1 snapshots and the storage and commit-time projections of Appendix D.3.

Reads only logs/e1/*/attempt-*/units.json (field snapshot.size_rw in bytes, snapshot.seconds, deviation_step).
No E7 path is opened (asserted). Run: python -B -I snapshot_cost_strata.py [harness logs root]
"""
import glob
import json
import sys

import os
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
default_logs = os.environ.get("AGENT_RISK_LOGS", os.path.join(REPO_ROOT, "data", "logs"))
H = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else default_logs
step0, later = [], []
for f in sorted(glob.glob(H.rstrip("/") + "/e1/*/attempt-*/units.json")):
    assert "/e7" not in f, f
    for u in json.load(open(f))["units"]:
        sn = u["snapshot"]
        (step0 if u["deviation_step"] == 0 else later).append((sn["size_rw"], sn["seconds"]))
assert len(step0) + len(later) == 485, (len(step0), len(later))
mb0 = sum(a for a, _ in step0) / len(step0)
mb1 = sum(a for a, _ in later) / len(later)
s0 = sum(b for _, b in step0) / len(step0)
s1 = sum(b for _, b in later) / len(later)
print(f"step 0: {len(step0)} snapshots, mean {mb0 / 1e6:.2f} MB, {s0:.2f} s")
print(f"after step 0: {len(later)} snapshots, mean {mb1 / 1e6:.2f} MB, {s1:.2f} s")
for name, total, at0 in (("theta_min 0.80 rule", 1625, 353), ("theta_min 0.85 to 0.94 rule", 1548, 317),
                         ("records after step 0", 213, 0)):
    gb = (at0 * mb0 + (total - at0) * mb1) / 1e9
    mins = (at0 * s0 + (total - at0) * s1) / 60
    print(f"{name}: {total} snapshots ({at0} at step 0): {gb:.1f} GB, {mins:.1f} minutes")
