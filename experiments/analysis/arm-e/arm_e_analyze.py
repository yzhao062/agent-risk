"""Arm E analysis: agreement mass, first-decision bound, and per-step hazards per rung.

  python arm_e_analyze.py --results results --data data --out results/analysis

For every condition (a rung replayed on one logged source) and every interface
(raw, full, full_ws, action; see arm_e_common.py), with one episode per logged
sample that Inspect scored:

  agreement mass   share of episodes whose every logged decision is reproduced
                   (Corollary 3.4: the identified agreement mass at that interface;
                   only the full interface identifies it, because later kernels read
                   the text as well as the tool calls);
  first decision   share of episodes whose first decision is reproduced, an upper
                   bound on the agreement mass;
  hazards          for step j, N_j episodes agree through step j-1, A_j of them make
                   decision j (the rest already ended), D_j disagree at step j.
                   Clock hazard 1 - D_j/N_j (its product over j equals the mass) and
                   per-decision agreement 1 - D_j/A_j;
  unconditional    (all-steps runs) agreement at step j on the logged input whatever
                   happened before, a diagnostic.

Intervals: exact Clopper-Pearson 95% for proportions, and 95% percentile intervals
from 10,000 episode-level bootstrap resamples (the bootstrap interval collapses to
a point when every episode has the same outcome; the exact interval does not).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import beta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_e_common import LEVELS, agree, first_char_divergence  # noqa: E402

B = 10_000
MAX_STEPS = 15

# rung, source, description, glob (relative to --results)
CONDITIONS = [
    ("r1_same", "bi_baseline", "Qwen3-14B-FP8, batch-invariant kernels, 32 in flight (as logged)", "r1_same-*/bi_all.jsonl"),
    ("r1_same", "c1_baseline", "Qwen3-14B-FP8, default kernels, one at a time (as logged)", "r1_same-*/c1_all_default.jsonl"),
    ("r1k_kernel", "bi_baseline", "Qwen3-14B-FP8, default kernels one at a time, vs the batch-invariant log (early stop)", "r1_same-*/bi_early_default.jsonl"),
    ("r1k_kernel", "c1_baseline", "Qwen3-14B-FP8, batch-invariant kernels, vs the default-kernel log", "r1_same-*/c1_all_bikernels.jsonl"),
    ("r2_bf16", "bi_baseline", "Qwen3-14B bf16, batch-invariant kernels", "r2_bf16-*/bi_all.jsonl"),
    ("r2_bf16", "c1_baseline", "Qwen3-14B bf16, batch-invariant kernels", "r2_bf16-*/c1_all.jsonl"),
    ("r3_qwen35", "bi_baseline", "Qwen3.5-27B-FP8, one at a time", "r3_qwen35-*/task*/bi_shard.jsonl"),
    ("r3_qwen35", "c1_baseline", "Qwen3.5-27B-FP8, one at a time", "r3_qwen35-*/task*/c1_shard.jsonl"),
    ("r4_qwen25", "bi_baseline", "Qwen2.5-14B-Instruct, batch-invariant kernels", "r4_qwen25-*/bi_all.jsonl"),
    ("r4_qwen25", "c1_baseline", "Qwen2.5-14B-Instruct, batch-invariant kernels", "r4_qwen25-*/c1_all.jsonl"),
]

PROBES = [
    ("r1_same", "r1_same-*/probe_bi.jsonl", "r1_same-*/bi_all.jsonl r1_same-*/c1_all_bikernels.jsonl"),
    ("r1_same_default", "r1_same-*/probe_default.jsonl", "r1_same-*/c1_all_default.jsonl r1_same-*/bi_early_default.jsonl"),
    ("r2_bf16", "r2_bf16-*/probe.jsonl", "r2_bf16-*/bi_all.jsonl r2_bf16-*/c1_all.jsonl"),
    ("r4_qwen25", "r4_qwen25-*/probe.jsonl", "r4_qwen25-*/bi_all.jsonl r4_qwen25-*/c1_all.jsonl"),
    ("r3_qwen35", "r3_qwen35-*/task*/probe.jsonl", "r3_qwen35-*/task*/bi_shard.jsonl r3_qwen35-*/task*/c1_shard.jsonl"),
]


def cp(k: int, n: int) -> list | None:
    if n == 0:
        return None
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return [round(lo, 4), round(hi, 4)]


def boot(x: np.ndarray, rng: np.random.Generator) -> list | None:
    if len(x) == 0:
        return None
    idx = rng.integers(0, len(x), size=(B, len(x)))
    m = x[idx].mean(axis=1)
    return [round(float(np.quantile(m, 0.025)), 4), round(float(np.quantile(m, 0.975)), 4)]


def load_logged(data: str) -> tuple[dict, dict]:
    logged, episodes = {}, {}
    for s in ("bi_baseline", "c1_baseline"):
        with open(os.path.join(data, f"logged_{s}.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                logged[(s, r["sample_id"], r["step"])] = r
                if r["episode_ok"]:
                    episodes.setdefault(s, {})[r["sample_id"]] = r["n_steps"]
    return logged, episodes


def load_rows(results: str, patterns: str) -> tuple[dict, list]:
    rows, files = {}, []
    for pat in patterns.split():
        for p in sorted(glob.glob(os.path.join(results, pat))):
            files.append(os.path.relpath(p, results))
            with open(p) as f:
                for line in f:
                    r = json.loads(line)
                    if r.get("status") != "decision":
                        continue
                    rows.setdefault((r["source"], r["sample_id"], r["step"]), r)
    return rows, files


def analyze_condition(rung, source, desc, pattern, results, logged, episodes, rng) -> dict | None:
    rows, files = load_rows(results, pattern)
    if not rows:
        return None
    eps = episodes[source]
    out = {"rung": rung, "source": source, "description": desc, "files": files,
           "n_steps_replayed": sum(1 for k in rows if k[0] == source)}
    per_ep = {}
    for sid, n in sorted(eps.items()):
        verdicts = []
        for j in range(n):
            r = rows.get((source, sid, j))
            if r is None:
                verdicts.append(None)
                continue
            verdicts.append(agree_levels := {lv: agree(logged[(source, sid, j)]["logged"], r["replay"], lv) for lv in LEVELS})
        per_ep[sid] = (n, verdicts)
    out["levels"] = {}
    for lv in LEVELS:
        mass, first, usable, incomplete = [], [], [], 0
        first_dis = {}
        for sid, (n, vs) in per_ep.items():
            fd = None
            complete = True
            for j, v in enumerate(vs):
                if v is None:
                    complete = False
                    break
                if not v[lv]:
                    fd = j
                    break
            if fd is None and not complete:
                incomplete += 1
                continue
            usable.append(sid)
            first_dis[sid] = fd
            mass.append(1 if fd is None else 0)
            first.append(0 if fd == 0 else 1)
        mass_a, first_a = np.array(mass, dtype=float), np.array(first, dtype=float)
        k, n = int(mass_a.sum()), len(mass_a)
        k1 = int(first_a.sum())
        haz = []
        for j in range(MAX_STEPS):
            N = [s for s in usable if first_dis[s] is None or first_dis[s] >= j]
            A = [s for s in N if per_ep[s][0] > j]
            D = [s for s in A if first_dis[s] == j]
            haz.append({"step": j + 1, "N_agree_so_far": len(N), "A_deciding": len(A), "D_disagree": len(D),
                        "clock_agreement": None if not N else round(1 - len(D) / len(N), 4),
                        "per_decision_agreement": None if not A else round(1 - len(D) / len(A), 4),
                        "per_decision_cp95": cp(len(A) - len(D), len(A))})
        # bootstrap the per-decision agreement at each step and the survival product
        ids = np.array(usable)
        fdv = np.array([MAX_STEPS + 1 if first_dis[s] is None else first_dis[s] for s in usable])
        nst = np.array([per_ep[s][0] for s in usable])
        bidx = rng.integers(0, len(ids), size=(B, len(ids))) if len(ids) else None
        for h in haz:
            j = h["step"] - 1
            if bidx is None:
                continue
            fdb, nsb = fdv[bidx], nst[bidx]
            Nb = (fdb >= j)
            Ab = Nb & (nsb > j)
            Db = Ab & (fdb == j)
            a_cnt, d_cnt = Ab.sum(axis=1), Db.sum(axis=1)
            ok = a_cnt > 0
            if ok.sum() >= 0.5 * B:
                v = 1 - d_cnt[ok] / a_cnt[ok]
                h["per_decision_boot95"] = [round(float(np.quantile(v, 0.025)), 4), round(float(np.quantile(v, 0.975)), 4)]
        # unconditional per-step agreement (every replayed step, whatever came before)
        uncond = []
        for j in range(MAX_STEPS):
            vals = [vs[j][lv] for (n_, vs) in per_ep.values() if j < len(vs) and vs[j] is not None]
            if vals:
                uncond.append({"step": j + 1, "n": len(vals), "agree": sum(vals), "rate": round(sum(vals) / len(vals), 4)})
        all_steps = [v[lv] for (n_, vs) in per_ep.values() for v in vs if v is not None]
        out["levels"][lv] = {
            "n_episodes": n, "n_incomplete": incomplete,
            "agreement_mass": round(k / n, 4) if n else None, "mass_k": k,
            "mass_cp95": cp(k, n), "mass_boot95": boot(mass_a, rng),
            "first_decision": round(k1 / n, 4) if n else None, "first_k": k1,
            "first_cp95": cp(k1, n), "first_boot95": boot(first_a, rng),
            "hazards": haz, "unconditional_by_step": uncond,
            "steps_replayed": len(all_steps), "steps_agree": int(sum(all_steps)),
        }
    # disagreement examples for the strictest same-model check
    ex = []
    for (src, sid, j), r in rows.items():
        if src != source:
            continue
        lg = logged[(src, sid, j)]["logged"]
        if not agree(lg, r["replay"], "raw") and len(ex) < 25:
            ex.append({"sample_id": sid, "step": j + 1, "first_char_divergence": first_char_divergence(lg, r["replay"]),
                       "agree": {lv: agree(lg, r["replay"], lv) for lv in LEVELS},
                       "logged_tools": [t[0] for t in lg.get("tool_calls", [])] if lg["kind"] == "msg" else lg["kind"],
                       "replay_tools": [t[0] for t in r["replay"].get("tool_calls", [])] if r["replay"]["kind"] == "msg" else r["replay"].get("error_class"),
                       "logged_content_len": len(lg.get("content") or ""), "replay_content_len": len(r["replay"].get("content") or ""),
                       "replay_finish": r["replay"].get("finish_reason")})
    out["raw_disagreement_examples"] = sorted(ex, key=lambda e: (e["sample_id"], e["step"]))
    # unconditional step agreement by the logged decision's length in tokens
    buckets = [(0, 50), (51, 100), (101, 200), (201, 400), (401, 10 ** 6)]
    by_len = []
    for lo, hi in buckets:
        sel = [(k_, r) for k_, r in rows.items() if k_[0] == source and k_[1] in eps
               and lo <= (logged[k_]["logged_completion_tokens"] or 0) <= hi and logged[k_]["logged"]["kind"] == "msg"]
        if sel:
            by_len.append({"logged_tokens": f"{lo}-{hi if hi < 10 ** 6 else 'max'}", "n": len(sel),
                           **{lv: sum(1 for k_, r in sel if agree(logged[k_]["logged"], r["replay"], lv)) for lv in ("full", "action")},
                           "first_steps": sum(1 for k_, _ in sel if k_[2] == 0)})
    out["agreement_by_logged_length"] = by_len
    # descriptive: how often the replay writes text, and its decode length
    reps = [r for k_, r in rows.items() if k_[0] == source]
    out["replay_profile"] = {
        "steps": len(reps),
        "with_text_content": sum(1 for r in reps if r["replay"].get("content")),
        "with_reasoning": sum(1 for r in reps if r["replay"].get("reasoning")),
        "errors": sum(1 for r in reps if r["replay"]["kind"] == "error"),
        "no_tool_call": sum(1 for r in reps if r["replay"]["kind"] == "msg" and not r["replay"]["tool_calls"]),
        "finish_length": sum(1 for r in reps if r["replay"].get("finish_reason") == "length"),
        "mean_completion_tokens": round(float(np.mean([(r.get("usage") or {}).get("completion_tokens") or 0 for r in reps])), 1) if reps else None,
    }
    return out


def analyze_probe(name, probe_pat, main_pats, results, logged) -> dict | None:
    files = sorted(glob.glob(os.path.join(results, probe_pat)))
    if not files:
        return None
    main, _ = load_rows(results, main_pats)
    per_file = {}
    for p in files:
        rows = {}
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                if r.get("status") == "decision":
                    rows[(r["source"], r["sample_id"], r["step"])] = r
        per_file[os.path.relpath(p, results)] = rows
    out = {"name": name, "files": list(per_file)}
    # probe vs main pass on the same server (main at its own concurrency, probe one at a time)
    same, tok_same, tok_n, n = 0, 0, 0, 0
    for fname, rows in per_file.items():
        mains = {}
        # restrict main rows to the same run directory as this probe file
        d = os.path.dirname(fname)
        for pat in main_pats.split():
            for p in glob.glob(os.path.join(results, d, os.path.basename(pat))):
                with open(p) as f:
                    for line in f:
                        r = json.loads(line)
                        if r.get("status") == "decision":
                            mains.setdefault((r["source"], r["sample_id"], r["step"]), r)
        for k, r in rows.items():
            if k in mains:
                n += 1
                same += agree(r["replay"], mains[k]["replay"], "raw")
                if r.get("token_ids") is not None and mains[k].get("token_ids") is not None:
                    tok_n += 1
                    tok_same += r["token_ids"] == mains[k]["token_ids"]
    out["probe_vs_main_same_server"] = {"n": n, "raw_identical": same, "token_ids_identical": tok_same,
                                        "token_ids_compared": tok_n}
    # probe across files (different GPUs and nodes)
    names = list(per_file)
    if len(names) > 1:
        keys = set.intersection(*[set(per_file[f]) for f in names])
        ident = sum(1 for k in keys if all(per_file[f][k].get("token_ids") == per_file[names[0]][k].get("token_ids")
                                           and agree(per_file[f][k]["replay"], per_file[names[0]][k]["replay"], "raw")
                                           for f in names))
        out["probe_across_runs"] = {"n_runs": len(names), "n_steps": len(keys), "identical_in_all_runs": ident}
    # probe vs the log
    allrows = per_file[names[0]]
    out["probe_vs_log_first_file"] = {
        "n": len(allrows),
        **{lv: sum(1 for k, r in allrows.items() if agree(logged[k]["logged"], r["replay"], lv)) for lv in LEVELS}}
    return out


def fmt_ci(ci) -> str:
    return "n/a" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def tables(res: dict) -> str:
    L = []
    L.append("## Agreement Mass and First-Decision Bound\n")
    L.append("| Rung | Source | Interface | Episodes | Agreement mass | Exact 95% | Bootstrap 95% | First decision | Exact 95% | Bootstrap 95% |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for c in res["conditions"]:
        for lv in ("full", "action", "raw", "full_ws"):
            d = c["levels"][lv]
            if not d["n_episodes"]:
                L.append(f"| {c['rung']} | {c['source']} | {lv} | 0 (incomplete {d['n_incomplete']}) | | | | | | |")
                continue
            L.append(f"| {c['rung']} | {c['source']} | {lv} | {d['n_episodes']} | {d['mass_k']}/{d['n_episodes']} = {d['agreement_mass']:.3f} | "
                     f"{fmt_ci(d['mass_cp95'])} | {fmt_ci(d['mass_boot95'])} | {d['first_k']}/{d['n_episodes']} = {d['first_decision']:.3f} | "
                     f"{fmt_ci(d['first_cp95'])} | {fmt_ci(d['first_boot95'])} |")
    L.append("\n## Per-Step Hazards (bi_baseline)\n")
    for c in res["conditions"]:
        if c["source"] != "bi_baseline":
            continue
        for lv in ("full", "action"):
            L.append(f"\n**{c['rung']}, {lv} interface** ({c['description']})\n")
            L.append("| Step | Agree so far N | Deciding A | Disagree D | Per-decision agreement 1-D/A | Exact 95% | Bootstrap 95% | Clock agreement 1-D/N | Unconditional agreement |")
            L.append("|---|---|---|---|---|---|---|---|---|")
            unc = {u["step"]: u for u in c["levels"][lv]["unconditional_by_step"]}
            for h in c["levels"][lv]["hazards"]:
                if h["N_agree_so_far"] == 0 and h["step"] > 1:
                    break
                u = unc.get(h["step"])
                ustr = "" if u is None else f"{u['agree']}/{u['n']} = {u['rate']:.3f}"
                pda = "n/a" if h["per_decision_agreement"] is None else f"{h['per_decision_agreement']:.3f}"
                ca = "n/a" if h["clock_agreement"] is None else f"{h['clock_agreement']:.3f}"
                L.append(f"| {h['step']} | {h['N_agree_so_far']} | {h['A_deciding']} | {h['D_disagree']} | {pda} | "
                         f"{fmt_ci(h['per_decision_cp95'])} | {fmt_ci(h.get('per_decision_boot95'))} | {ca} | {ustr} |")
    L.append("\n## Step Agreement by Logged Decision Length (all replayed steps, whatever came before)\n")
    L.append("| Rung | Source | Logged tokens | Steps | First steps among them | Full agree | Action agree |")
    L.append("|---|---|---|---|---|---|---|")
    for c in res["conditions"]:
        for b in c.get("agreement_by_logged_length", []):
            L.append(f"| {c['rung']} | {c['source']} | {b['logged_tokens']} | {b['n']} | {b['first_steps']} | "
                     f"{b['full']} ({b['full'] / b['n']:.3f}) | {b['action']} ({b['action'] / b['n']:.3f}) |")
    L.append("\n## Replay Profile\n")
    L.append("| Rung | Source | Steps replayed | Full agree | Action agree | With text | With reasoning | Errors | No tool call | Hit max_tokens | Mean output tokens |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in res["conditions"]:
        p, f_, a_ = c["replay_profile"], c["levels"]["full"], c["levels"]["action"]
        L.append(f"| {c['rung']} | {c['source']} | {p['steps']} | {f_['steps_agree']}/{f_['steps_replayed']} | {a_['steps_agree']}/{a_['steps_replayed']} | "
                 f"{p['with_text_content']} | {p['with_reasoning']} | {p['errors']} | {p['no_tool_call']} | {p['finish_length']} | {p['mean_completion_tokens']} |")
    L.append("\n## Determinism Probes\n")
    for p in res["probes"]:
        L.append(f"- {p['name']}: probe vs main pass on the same server {p['probe_vs_main_same_server']}; "
                 f"across runs {p.get('probe_across_runs')}; probe vs log {p['probe_vs_log_first_file']}")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(20261004)
    logged, episodes = load_logged(a.data)
    res = {"bootstrap_resamples": B, "conditions": [], "probes": []}
    for rung, src, desc, pat in CONDITIONS:
        c = analyze_condition(rung, src, desc, pat, a.results, logged, episodes, rng)
        if c is not None:
            res["conditions"].append(c)
    for name, ppat, mpats in PROBES:
        p = analyze_probe(name, ppat, mpats, a.results, logged)
        if p is not None:
            res["probes"].append(p)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "analysis.json"), "w") as f:
        json.dump(res, f, indent=1)
    t = tables(res)
    with open(os.path.join(a.out, "tables.md"), "w") as f:
        f.write(t)
    print(t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
