"""Extract E2 episode records (extract_episodes.episode_record, unchanged, plus
the per-sample gate fields that an E2 eval carries in sample metadata).

extract_episodes.py takes gate and theta from the task metadata, which in an E2
eval reads "per-sample"; this wrapper overrides them with the sample's own
``arh_gate`` and the block's (lam, pi), and adds the condition, the pair index,
the tape values the live gate recorded, and per step the monitor record (the gate
score s the gate read, E[N]/100, its tail, the discrete score, the status and the
probe tree). The per-step "score" fields of the base record are suspicion_score,
which in E2 is the gate score s = -log10(1 - E[N]/100), capped at 15.

  python e2_extract.py --log-dir <attempt dir> --out <attempt dir>/episodes.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from inspect_ai.log import read_eval_log
from inspect_ai.model import ChatMessageAssistant

from extract_episodes import episode_record


def e2_record(sample, log_meta: dict) -> dict:
    rec = episode_record(sample, log_meta)
    md = sample.metadata or {}
    gate = md.get("arh_gate")
    rec["gate"] = gate
    rec["theta"] = log_meta.get("e2_lam") if gate == "stopping" else None
    rec["e2"] = {"condition": md.get("arh_condition"), "pair": md.get("arh_pair"), "gate": gate,
                 "lam": log_meta.get("e2_lam"), "pi": log_meta.get("e2_pi"),
                 "block": (log_meta.get("e2_block") or {}).get("block"),
                 "host": (log_meta.get("e2_block") or {}).get("host")}
    if rec.get("scored"):
        rec["e2"]["gate_u"] = [{"step": d.get("step"), "u": d.get("e2_u"), "decision": d.get("decision")}
                               for d in (rec.get("gate_decisions") or []) if d.get("e2_u") is not None]
    # per-step monitor record: s (the score the gate read), E[N]/100, the tail, the discrete score, the tree
    mon = []
    k = 0
    for m in sample.messages:
        if not isinstance(m, ChatMessageAssistant):
            continue
        mm = m.metadata or {}
        det = mm.get("e2_cscore_detail") or {}
        mon.append({"step": mm.get("arh_step", k), "score": mm.get("suspicion_score"),
                    "discrete": mm.get("e2_score_discrete"), "cscore": mm.get("e2_cscore"),
                    "tail": mm.get("e2_tail"), "doubt": mm.get("e2_doubt"),
                    "status": mm.get("e2_cscore_status"), "gate_score_source": mm.get("e2_gate_score_source"),
                    "detail": {key: det.get(key) for key in ("expected_raw", "tail", "valid_mass", "invalid_mass",
                                                              "unexplored_mass", "bounds", "n_probes", "greedy",
                                                              "level1_digits", "probes", "usage", "messages_source",
                                                              "messages_sha", "tail_bounds", "extra_first_digits",
                                                              "eps", "rel")} if det else None})
        k += 1
    rec["e2"]["monitor"] = mon
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(os.path.expanduser(args.log_dir), "*.eval")))
    n = 0
    with open(os.path.expanduser(args.out), "w") as f:
        for p in paths:
            log = read_eval_log(p, resolve_attachments=True)
            meta = log.eval.metadata or {}
            for s in log.samples or []:
                f.write(json.dumps(e2_record(s, meta), default=str) + "\n")
                n += 1
    print(f"wrote {n} episodes from {len(paths)} log(s) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
