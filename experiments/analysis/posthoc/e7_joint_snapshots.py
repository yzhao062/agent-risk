"""Post hoc: snapshots needed when the query reads harm and task success together. A continuation can change
success at every running maximum, so no record is closed; step-0 records start from the initial state."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
LOGS_DEFAULT = Path(os.environ.get("AGENT_RISK_LOGS", str(REPO / "data" / "logs"))) / "e7"
MANIFEST_DEFAULT = REPO / "data" / "manifests" / "e7" / "manifest_e7.json"
ANALYZE_E7_PATH = REPO / "experiments" / "analysis" / "analyze_e7.py"
RP_PATH = HERE / "e7_records_posthoc.py"

parser = argparse.ArgumentParser(description="Snapshots needed when query reads harm and task success together")
parser.add_argument("--logs", default=str(LOGS_DEFAULT), help="Path to logs/e7 directory")
parser.add_argument("--manifest", default=str(MANIFEST_DEFAULT), help="Path to manifest_e7.json")
parser.add_argument("--analyze-e7", default=str(ANALYZE_E7_PATH), help="Path to analyze_e7.py")
parser.add_argument("--e7-records-posthoc", default=str(RP_PATH), help="Path to e7_records_posthoc.py")
parser.add_argument("--out", default=str(HERE / "posthoc_joint_snapshots_e7.json"), help="Output path")
args = parser.parse_args()

spec = importlib.util.spec_from_file_location("rp", args.e7_records_posthoc)
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)

LAST = 14   # 15-step cap: a record at step 14 leaves no agent step, so its harm is fixed and it is not open for harm
spec2 = importlib.util.spec_from_file_location("analyze_e7", args.analyze_e7)
a7 = importlib.util.module_from_spec(spec2)
sys.path.insert(0, str(Path(args.analyze_e7).parent))
spec2.loader.exec_module(a7)

manifest = json.load(open(args.manifest))
recs, _ = a7.collect_records(args.logs, manifest, ("base",))
eps = [v for v in recs.values() if v["condition"] == "base"]
tot = {"all": [0, 0.0], "records": [0, 0.0], "records_after_step0": [0, 0.0], "open_harm_after_step0": [0, 0.0]}
for e in eps:
    rec = set(rp.records(e, rp.FLOOR_E7))
    for s in (e.get("e7") or {}).get("snapshots") or []:
        c = s.get("commit") or {}
        if not c.get("ok"):
            continue
        t, b = int(s["step"]), float(c.get("size_rw") or 0)
        keys = ["all"] + (["records"] if t in rec else []) + (["records_after_step0"] if t in rec and t > 0 else []) \
            + (["open_harm_after_step0"] if t in rec and 0 < t < LAST and not rp.harm_before(e, t) else [])
        for k in keys:
            tot[k][0] += 1
            tot[k][1] += b
out = {k: {"snapshots": n, "gb": round(b / 1e9, 3)} for k, (n, b) in tot.items()}
out["note"] = ("Post hoc (not registered). Committed snapshots of the base (pass-all) E7 episodes. 'records': running maxima of "
               "the monitor score; 'records_after_step0': those after step 0 (a step-0 record restarts from the initial state); "
               "'open_harm_after_step0': those with no harm earlier in the episode and an agent step left (not step 14 of the 15-step cap). A query that reads harm and task success "
               "together keeps every record after step 0.")
with open(args.out, "w") as f:
    f.write(json.dumps(out, indent=2) + "\n")
print(json.dumps(out))
