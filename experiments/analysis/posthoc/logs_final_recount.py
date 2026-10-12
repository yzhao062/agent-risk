"""Recount of the log-derived numbers that the final Appendix D.3 paragraph quotes.

Reads only logs/e1/*/attempt-*/episodes_base.jsonl (pass-all log), logs/d58 (replication log, gate == baseline)
and logs/e2 (stronger agent, gate == baseline). No path containing "e7" is opened (asserted).
A step is a decision when its score is not None and it proposed at least one call (the harness rule of
analyze_pilot.abort_step). A gate with threshold theta refuses the first decision with score > theta; a record
above a floor is a decision whose score exceeds the floor and every earlier decision score.
Run: python -B -I logs_final_recount.py
"""
import glob
import json
import sys
from collections import Counter

import os
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
H = os.environ.get("AGENT_RISK_LOGS", os.path.join(REPO_ROOT, "data", "logs"))
if not H.endswith("/"):
    H += "/"
if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
    H = sys.argv[1]
    if not H.endswith("/"):
        H += "/"
FAILS = []


def check(name, got, want):
    ok = got == want
    print(("PASS " if ok else "FAIL ") + f"{name}: got {got!r}, quoted {want!r}")
    if not ok:
        FAILS.append(name)


def load(pattern, gate=None):
    eps, seen = [], set()
    for f in sorted(glob.glob(H + pattern)):
        assert "e7" not in f.split("/logs/")[1].split("/")[0], f
        for line in open(f):
            if not line.strip():
                continue
            d = json.loads(line)
            if gate is not None and d.get("gate") != gate:
                continue
            if not d.get("per_step"):
                continue
            key = d.get("unit_id") or d.get("sample_id")
            assert key not in seen, key
            seen.add(key)
            eps.append(d)
    return eps


def dec(ep):
    return [(r["step"], r["score"]) for r in sorted(ep["per_step"], key=lambda r: r["step"])
            if r["score"] is not None and r["n_proposed_calls"] > 0]


def records(ep, floor):
    out, m = [], None
    for t, x in dec(ep):
        top = floor if m is None else max(floor, m)
        if x > top:
            out.append(t)
        m = x if m is None else max(m, x)
    return out


def first_block(ep, theta):
    for t, x in dec(ep):
        if x > theta:
            return t
    return None


def harm_before(ep, t):
    return any(r["violation"] and r["step"] < t for r in ep["per_step"])


e1 = load("e1/*/attempt-*/episodes_base.jsonl")
n = len(e1)
check("E1 episodes", n, 1200)
last = max(r["step"] for e in e1 for r in e["per_step"])
check("E1 last step index", last, 14)

R80 = [(e, t) for e in e1 for t in records(e, 0.80)]
R94 = [(e, t) for e in e1 for t in records(e, 0.94)]
check("records above 0.80", len(R80), 566)
check("records above 0.94", len(R94), 520)
check("records above 0.80 with harm before (closed)", sum(harm_before(e, t) for e, t in R80), 9)
check("records above 0.94 with harm before (closed)", sum(harm_before(e, t) for e, t in R94), 9)
check("open records above 0.94 (511)", sum(not harm_before(e, t) for e, t in R94), 511)
check("records above 0.80 at step 0", sum(t == 0 for e, t in R80), 353)
check("records above 0.80 after step 0 (at most 213 need a snapshot)", sum(t > 0 for e, t in R80), 213)
check("open records above 0.80 after step 0", sum(t > 0 and not harm_before(e, t) for e, t in R80), 204)
check("records above 0.80 at the last step index", sum(t == last for e, t in R80), 0)
check("records above 0.94 at the last step index", sum(t == last for e, t in R94), 0)

fb = [(e, first_block(e, 0.94)) for e in e1]
fb = [(e, t) for e, t in fb if t is not None]
check("first blocks at 0.94", len(fb), 485)
check("first blocks at 0.94 at the last step index", sum(t == last for e, t in fb), 0)
check("open first blocks at 0.94 (479)", sum(not harm_before(e, t) for e, t in fb), 479)

rep = load("d58/block-*/attempt-*/episodes.jsonl", gate="baseline")
check("replication episodes", len(rep), 1000)
rr = sum(len(records(e, 0.80)) for e in rep)
rs = sum(sum(x > 0.80 for _, x in dec(e)) for e in rep)
check("replication records above 0.80", rr, 455)
check("replication steps above 0.80", rs, 1160)
check("replication records per episode, exact tie 0.455 -> 0.46", rr * 1000 == 455 * len(rep), True)

e2 = load("e2/*/attempt-*/episodes.jsonl", gate="baseline")
check("stronger-agent episodes", len(e2), 1399)
lam = 11.551
r2 = sum(len(records(e, lam)) for e in e2)
s2 = sum(sum(x > lam for _, x in dec(e)) for e in e2)
check("stronger-agent records above lambda=11.551", r2, 255)
check("stronger-agent steps above lambda=11.551", s2, 333)
check("records per episode 0.18", round(r2 / len(e2), 2), 0.18)
check("steps per episode 0.24", round(s2 / len(e2), 2), 0.24)
print("INFO e2 score range", min(x for e in e2 for _, x in dec(e)), max(x for e in e2 for _, x in dec(e)))

print("\nFAILED:" if FAILS else "\nALL NUMBERS REPRODUCE", FAILS if FAILS else "")
sys.exit(1 if FAILS else 0)
