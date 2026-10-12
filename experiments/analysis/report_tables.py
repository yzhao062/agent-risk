"""Render analysis_final.json (analyze_pilot.py) as Markdown tables.

Usage: python report_tables.py analysis_final.json > tables.md
"""

from __future__ import annotations

import json
import sys


def f(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, float) and abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def pci(d, nd=3):
    return f"{f(d['point'], nd)} [{f(d['ci95'][0], nd)}, {f(d['ci95'][1], nd)}]"


def main() -> int:
    r = json.load(open(sys.argv[1]))
    out = []
    out.append(f"B0 episodes scored: {r['n_b0']} (unscored: {r['n_b0_unscored']})\n")
    out.append("| Query (B0, baseline gate) | Point [95% CI] |\n|---|---|")
    for k in ("p_harm", "p_failure", "mean_total_tokens", "cvar90_total_tokens",
              "mean_agent_input_tokens", "mean_agent_output_tokens", "mean_monitor_input_tokens",
              "mean_monitor_output_tokens", "mean_tool_calls",
              "p_harm_egress", "p_harm_fs_write", "p_harm_destructive", "p_harm_privilege"):
        out.append(f"| {k} | {pci(r['baseline'][k], 4 if k.startswith('p_') else 1)} |")
    out.append("")
    out.append(f"Baseline extra: {json.dumps(r['baseline_extra'])}\n")
    sd = r["score_distribution"]
    out.append(f"Scored steps: {sd['n_scored_steps']}; step score counts: {sd['step_score_counts']}\n")
    out.append(f"Episode max score counts: {sd['episode_max_score_counts']}\n")
    out.append("| Target | theta | Replayed abort rate | Equivalent theta interval |\n|---|---|---|---|")
    for g in r["theta_grid"]:
        out.append(f"| {g.get('target')} | {g['theta']} | {f(g['replay_abort_rate'])} | {g.get('equivalent_theta_interval')} |")
    out.append("")

    # arm A
    keys = ("abort_rate", "p_harm", "p_failure", "mean_total_tokens", "cvar90_total_tokens", "mean_tool_calls")
    for t, rep in r["arm_a_replay"].items():
        val = r.get("arm_a_validation2", {}).get(t)
        out.append(f"\n**theta = {t}**" + (f" (validation n = {val['n']}, unscored {val['n_unscored']})" if val else ""))
        out.append("\n| Query | Replay from B0 (n = %d) | Live stopping gate | Live minus replay |\n|---|---|---|---|" % r["n_b0"])
        for k in keys:
            nd = 1 if "tokens" in k or "calls" in k else 4
            live = pci(val["observed"][k], nd) if val else "n/a"
            diff = pci(val["observed_minus_replay"][k], nd) if val and val["observed_minus_replay"] else "n/a"
            out.append(f"| {k} | {pci(rep[k], nd)} | {live} | {diff} |")
        if val:
            out.append(f"\nControlArena partial-episode main-task score (diagnostic): overall {f(val['main_task_success_diag_overall'])}, aborted episodes {f(val['main_task_success_diag_aborted'])}; limits {val['limits']}")

    # arm B
    out.append("\n### Arm B\n")
    for t, b in r["arm_b"].items():
        out.append(f"\n**theta = {t}**\n")
        out.append("| Query | Full channel (point) | Coarse channel (identified interval) | Scores kept, per-step harm removed | Same, command checks recomputed |\n|---|---|---|---|---|")
        for k in ("p_harm", "p_failure", "mean_total_tokens", "cvar90_total_tokens", "mean_tool_calls", "abort_rate"):
            nd = 1 if "tokens" in k or "calls" in k else 4
            fc = f(b["full_channel_point"][k], nd)
            cc = b["coarse_channel_interval"][k]
            nh = b["no_step_harm_interval"][k]
            nc = b["no_step_harm_cmd_recomputed_interval"][k]
            out.append(f"| {k} | {fc} | [{f(cc['lo'], nd)}, {f(cc['hi'], nd)}] | [{f(nh['lo'], nd)}, {f(nh['hi'], nd)}] | [{f(nc['lo'], nd)}, {f(nc['hi'], nd)}] |")
        out.append(f"\nEpisodes whose harm-before-abort is unknown without per-step harm: {b['no_step_harm_interval']['n_episodes_harm_unknown']} (command checks recomputed: {b['no_step_harm_cmd_recomputed_interval']['n_episodes_harm_unknown']})")

    ex = r.get("arm_a_validation1")
    if ex:
        out.append("\n### Exactness (validation 1)\n")
        out.append({k: v for k, v in ex.items() if k != "rows"}.__repr__())
    c1 = r.get("concurrency1_diagnostic")
    if c1:
        out.append("\n### Concurrency 1 diagnostic\n")
        for k in ("stopping_c1_vs_baseline_c1", "stopping_c1_vs_b0"):
            out.append(f"{k}: " + repr({kk: vv for kk, vv in c1[k].items() if kk != "rows"}))
        out.append(f"baseline_c1_vs_b0: {c1['baseline_c1_vs_b0']}")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
