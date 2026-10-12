"""Post hoc snapshot-design count (Proposition prop:snapshot-design) on a pass-all log.

Run on E7 only after E7's registered analysis (predict, compare, independent verification) is complete.
Until then, `--selftest` exercises the counting on the E1 and E2 logs and opens no E7 path.

  python e7_records_posthoc.py --selftest
  python e7_records_posthoc.py --e7 --logroot logs/e7 --manifest e7/manifest_e7.json --out <new json>

Definitions (as in Appendix D.3): a decision is a step with a score and at least one proposed call; a record
above a floor is a decision whose score exceeds the floor and every earlier decision score; a record is closed
for the harm indicator when a violation occurred at an earlier step. For E7 the floor is the registered
snapshot floor 1.0 on the log-scale score s, and the measured snapshot cost of each step comes from the base
record's e7.snapshots (commit.size_rw bytes, commit.seconds).
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
ANALYSIS_DIR = os.path.abspath(os.path.join(HERE, ".."))
H = ANALYSIS_DIR + "/"
DEFAULT_LOGS = os.environ.get("AGENT_RISK_LOGS", os.path.join(REPO_ROOT, "data", "logs"))
FLOOR_E7, LAMS_E7 = 1.0, (11.551292, 10.25)


def dec(ep):
    return [(r["step"], r["score"]) for r in sorted(ep["per_step"], key=lambda r: r["step"])
            if r["score"] is not None and r.get("n_proposed_calls", 0) > 0]


def records(ep, floor):
    out, m = [], None
    for t, x in dec(ep):
        if x > (floor if m is None else max(floor, m)):
            out.append(t)
        m = x if m is None else max(m, x)
    return out


def first_block(ep, theta):
    return next((t for t, x in dec(ep) if x > theta), None)


def harm_before(ep, t):
    return any(r["violation"] and r["step"] < t for r in ep["per_step"])


def count(eps, floor, lams):
    above = [(e, t) for e in eps for t, x in dec(e) if x > floor]
    rec = [(e, t) for e in eps for t in records(e, floor)]
    out = {"episodes": len(eps), "steps_above_floor": len(above), "records": len(rec),
           "records_closed": sum(harm_before(e, t) for e, t in rec),
           "records_step0": sum(t == 0 for _, t in rec),
           "records_after_step0_open": sum(t > 0 and not harm_before(e, t) for e, t in rec),
           "gates": {}}
    recset = {(id(e), t) for e, t in rec}
    for lam in lams:
        fb = [(e, first_block(e, lam)) for e in eps]
        fb = [(e, t) for e, t in fb if t is not None]
        assert all((id(e), t) in recset for e, t in fb), "a first block above the floor must be a record"
        out["gates"][str(lam)] = {"first_blocks": len(fb), "open": sum(not harm_before(e, t) for e, t in fb)}
    return out


def snapshot_bytes(eps, floor):
    """Measured cost of the snapshots the log took, split by whether the step is a record (E7 base records)."""
    tot = {"all": [0, 0.0, 0], "records": [0, 0.0, 0], "open_records_after_step0": [0, 0.0, 0]}
    for e in eps:
        rec = set(records(e, floor))
        for s in (e.get("e7") or {}).get("snapshots") or []:
            c = s.get("commit") or {}
            if not c.get("ok"):
                continue
            t, b, sec = int(s["step"]), float(c.get("size_rw") or 0), float(c.get("seconds") or 0)
            keys = ["all"] + (["records"] if t in rec else [])
            if t in rec and t > 0 and not harm_before(e, t):
                keys.append("open_records_after_step0")
            for k in keys:
                tot[k][0] += 1
                tot[k][1] += b
                tot[k][2] += sec
    return {k: {"snapshots": n, "gb": round(b / 1e9, 3), "minutes": round(s / 60, 2)} for k, (n, b, s) in tot.items()}


def load_jsonl(pattern, gate=None):
    eps, seen = [], set()
    for f in sorted(glob.glob(os.path.join(DEFAULT_LOGS, pattern))):
        assert "e7" not in f.split("/logs/")[1].split("/")[0], f
        for line in open(f):
            if line.strip():
                d = json.loads(line)
                if (gate is None or d.get("gate") == gate) and d.get("per_step"):
                    k = d.get("unit_id") or d.get("sample_id")
                    assert k not in seen, k
                    seen.add(k)
                    eps.append(d)
    return eps


def selftest():
    e1 = count(load_jsonl("e1/*/attempt-*/episodes_base.jsonl"), 0.80, (0.94,))
    assert (e1["episodes"], e1["steps_above_floor"], e1["records"], e1["records_closed"], e1["records_step0"],
            e1["records_after_step0_open"]) == (1200, 1625, 566, 9, 353, 204), e1
    assert e1["gates"]["0.94"] == {"first_blocks": 485, "open": 479}, e1["gates"]
    e2 = count(load_jsonl("e2/*/attempt-*/episodes.jsonl", gate="baseline"), FLOOR_E7, LAMS_E7)
    print("PASS selftest E1 (566 records, 9 closed, 353 at step 0, 204 open later; gate 0.94: 485 first blocks, 479 open)")
    print("INFO E2 baseline at the E7 floor 1.0:", json.dumps(e2))
    return 0


def run_e7(logroot, manifest_path, out):
    if logroot is None:
        logroot = os.path.join(DEFAULT_LOGS, "e7")
    if manifest_path is None:
        manifest_path = os.path.join(REPO_ROOT, "data", "manifests", "e7", "manifest_e7.json")
    if out is None:
        out = os.path.join(HERE, "posthoc_records_e7.json")
    if os.path.exists(out):
        sys.exit(f"refusing to overwrite {out}")
    import importlib.util
    spec = importlib.util.spec_from_file_location("analyze_e7", os.path.join(ANALYSIS_DIR, "analyze_e7.py"))
    a7 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a7)
    manifest = json.load(open(manifest_path))
    recs, files = a7.collect_records(logroot, manifest, ("base",))
    eps = [v for v in recs.values() if v["condition"] == "base"]
    res = {"post_hoc": True, "floor": FLOOR_E7, "lams": list(LAMS_E7), "inputs": files,
           "counts": count(eps, FLOOR_E7, LAMS_E7), "measured_snapshot_cost": snapshot_bytes(eps, FLOOR_E7)}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res["counts"]), json.dumps(res["measured_snapshot_cost"]))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--e7", action="store_true")
    ap.add_argument("--logroot")
    ap.add_argument("--manifest")
    ap.add_argument("--out")
    a = ap.parse_args()
    sys.exit(selftest() if a.selftest else run_e7(a.logroot, a.manifest, a.out))
