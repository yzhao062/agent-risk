"""Render analysis_arms_cd.json and the retry_bounds outputs as Markdown tables.

Usage: python report_arms_cd.py analysis_arms_cd.json retry_bounds_b0.json [retry_bounds_b1.json ...] > tables_arms_cd.md
"""

from __future__ import annotations

import json
import sys

ROWS = [("p_harm", "P(harm)"), ("p_failure", "P(failure)"), ("abort_rate", "Abort rate"),
        ("mean_total_tokens", "Mean total tokens"), ("cvar90_total_tokens", "CVaR 0.9 of total tokens"),
        ("mean_agent_input_tokens", "Mean agent input tokens"), ("mean_agent_output_tokens", "Mean agent output tokens"),
        ("mean_monitor_input_tokens", "Mean monitor input tokens"), ("mean_monitor_output_tokens", "Mean monitor output tokens"),
        ("mean_tool_calls", "Mean executed tool calls"), ("p_harm_egress", "P(harm), egress"),
        ("p_harm_fs_write", "P(harm), file write"), ("p_harm_destructive", "P(harm), destructive"),
        ("p_harm_privilege", "P(harm), privilege")]


def f(x, k=""):
    if x is None:
        return "n/a"
    if k.startswith("p_") or k in ("abort_rate", "cap_abort_rate") or k.endswith("_mass") or k.startswith("harm") or k.startswith("failure"):
        return f"{x:.3f}"
    if abs(x) >= 100:
        return f"{x:,.0f}"
    return f"{x:.1f}"


def pci(d, k):
    return f"{f(d['point'], k)} [{f(d['ci95'][0], k)}, {f(d['ci95'][1], k)}]"


def dci(d, k):
    return f"{f(d['point'], k)} [{f(d['ci95'][0], k)}, {f(d['ci95'][1], k)}]"


def main() -> int:
    r = json.load(open(sys.argv[1]))
    bounds = {p: json.load(open(p)) for p in sys.argv[2:]}
    out = []
    t = r["theta"]
    if "arm_c" in r:
        c = r["arm_c"]
        out.append(f"## Arm C: retry gate at theta {t} (live n = {c['n']}, unscored {c['n_unscored']})\n")
        pc = c["rule2_primary_never_blocked_minus_063"]
        out.append("### Reading rule 2 (primary): never-blocked mass\n")
        out.append(f"Live {pc['k']}/{pc['n']} = {pc['p']:.3f} (Clopper-Pearson {pc['clopper_pearson_95'][0]:.3f} to {pc['clopper_pearson_95'][1]:.3f}); "
                   f"replay from B0 {pc['k_ref']}/{pc['n_ref']} = {pc['p_ref']:.3f}; difference {pc['diff']:.3f}, "
                   f"Newcombe [{pc['newcombe_95'][0]:.3f}, {pc['newcombe_95'][1]:.3f}], bootstrap [{pc['bootstrap_diff_ci95'][0]:.3f}, {pc['bootstrap_diff_ci95'][1]:.3f}].\n")
        sc = c["rule2_secondary_vs_val2_t2_never_aborted"]
        out.append(f"Secondary: live never-blocked {sc['p']:.3f} against val2_t2 never-aborted {sc['k_ref']}/{sc['n_ref']} = {sc['p_ref']:.3f}; "
                   f"difference {sc['diff']:.3f}, Newcombe [{sc['newcombe_95'][0]:.3f}, {sc['newcombe_95'][1]:.3f}].\n")
        sh = c["rule3_live_minus_delete_blocked"]
        out.append("### Reading rule 3: live P(harm) minus the delete-blocked shortcut\n")
        out.append(f"Shortcut from B0 {sh['delete_blocked_p_harm_b0']:.3f} [{sh['delete_blocked_ci95'][0]:.3f}, {sh['delete_blocked_ci95'][1]:.3f}]; live {sh['live_p_harm']:.3f}; "
                   f"difference {sh['live_minus_shortcut']:.3f}, bootstrap [{sh['bootstrap_diff_ci95'][0]:.3f}, {sh['bootstrap_diff_ci95'][1]:.3f}], "
                   f"Newcombe [{sh['exact']['newcombe_95'][0]:.3f}, {sh['exact']['newcombe_95'][1]:.3f}].\n")
        eh = c["exact_harm"]
        out.append(f"Exact: live P(harm) {eh['live_p_harm']['k']}/{eh['live_p_harm']['n']}, Clopper-Pearson [{eh['live_p_harm']['clopper_pearson_95'][0]:.3f}, {eh['live_p_harm']['clopper_pearson_95'][1]:.3f}]; "
                   f"P(harm and never blocked) live {eh['harm_and_agree_vs_replay_b0']['k']}/300 vs replay {eh['harm_and_agree_vs_replay_b0']['k_ref']}/300, "
                   f"Newcombe [{eh['harm_and_agree_vs_replay_b0']['newcombe_95'][0]:.3f}, {eh['harm_and_agree_vs_replay_b0']['newcombe_95'][1]:.3f}].\n")
        for label, rows in c["rule1_inside_outer_limits"].items():
            out.append(f"### Reading rule 1: live value inside the outer bootstrap limits ({label}, frozen file)\n")
            out.append("| Query | Outer limits | Live | Inside |\n|---|---|---|---|")
            for k, v in rows.items():
                out.append(f"| {k} | [{f(v['outer'][0], k)}, {f(v['outer'][1], k)}] | {f(v['live'], k)} | {'yes' if v['live_inside_outer_limits'] else 'NO'} |")
            out.append("")
        for label, rows in c.get("bounds_vs_live_stat", {}).items():
            out.append(f"### Bounds recomputed from {label}, with live minus lower end and upper end minus live\n")
            out.append("| Query | Bound | Live | Live minus lower [95%] | Upper minus live [95%] | Consistent at 95% |\n|---|---|---|---|---|---|")
            for k, v in rows.items():
                out.append(f"| {k} | [{f(v['bound'][0], k)}, {f(v['bound'][1], k)}] | {f(v['live'], k)} | "
                           f"{f(v['live_minus_lo'][0], k)} [{f(v['live_minus_lo'][1], k)}, {f(v['live_minus_lo'][2], k)}] | "
                           f"{f(v['hi_minus_live'][0], k)} [{f(v['hi_minus_live'][1], k)}, {f(v['hi_minus_live'][2], k)}] | {'yes' if v['consistent_at_95'] else 'NO'} |")
            out.append("")
        out.append("Monitor-input support of post-block proposals: " + json.dumps(c["monitor_input_support"]) + "\n")
        for label, chk in c["bounds_check"].items():
            out.append(f"### Live estimate versus declared bounds ({label})\n")
            out.append("| Query | Declared bound | Outer 95% limits | Live [95% CI] | Live point inside | Position in bound |\n|---|---|---|---|---|---|")
            for k, v in chk.items():
                lk = "abort_rate" if k == "cap_abort_rate" else k
                pos = "n/a" if v["position_in_bound"] is None else f"{v['position_in_bound']:.2f}"
                inside = "yes" if v["live_point_inside"] else "NO"
                out.append(f"| {k} | [{f(v['bound'][0], k)}, {f(v['bound'][1], k)}] | "
                           f"[{f(v['bound_outer_ci95'][0], k)}, {f(v['bound_outer_ci95'][1], k)}] | "
                           f"{f(v['live'], lk)} [{f(v['live_ci95'][0], lk)}, {f(v['live_ci95'][1], lk)}] | {inside} | {pos} |")
            out.append("")
        out.append("### Identified part (episodes never blocked)\n")
        out.append("| Quantity | Live [95% CI] | Replay from B0 [95% CI] | Live minus replay [95% CI] |\n|---|---|---|---|")
        for k, v in c["agreement_part"].items():
            out.append(f"| {k} | {f(v['live'], k)} [{f(v['live_ci95'][0], k)}, {f(v['live_ci95'][1], k)}] | "
                       f"{f(v['replay_b0'], k)} [{f(v['replay_b0_ci95'][0], k)}, {f(v['replay_b0_ci95'][1], k)}] | "
                       f"{f(v['live_minus_replay'], k)} [{f(v['diff_ci95'][0], k)}, {f(v['diff_ci95'][1], k)}] |")
        if any("replay_b1" in v for v in c["agreement_part"].values()):
            out.append("\n| Quantity | Replay from B1 [95% CI] | Live minus replay B1 [95% CI] |\n|---|---|---|")
            for k, v in c["agreement_part"].items():
                out.append(f"| {k} | {f(v['replay_b1'], k)} [{f(v['replay_b1_ci95'][0], k)}, {f(v['replay_b1_ci95'][1], k)}] | "
                           f"{f(v['live_minus_replay_b1'], k)} [{f(v['diff_b1_ci95'][0], k)}, {f(v['diff_b1_ci95'][1], k)}] |")
        out.append("\n### Retry statistics\n")
        out.append("```\n" + json.dumps(c["retry_stats"], indent=1) + "\n```\n")
        out.append("### Live retry gate versus live stopping gate (val2_t2) and stopping replay (B0)\n")
        out.append("| Query | Retry (live) | Stopping val2_t2 (live) | Retry minus stopping [95% CI] | Retry minus stopping replay [95% CI] |\n|---|---|---|---|---|")
        for k, lab in ROWS:
            out.append(f"| {lab} | {pci(c['observed'][k], k)} | {pci(r['val2_t2']['observed'][k], k)} | "
                       f"{dci(c['retry_minus_stopping_val2_t2'][k], k)} | {dci(c['retry_minus_stopping_replay_b0'][k], k)} |")
        out.append("")
    for path, b in bounds.items():
        out.append(f"### Declared bounds detail ({path.split('/')[-1]})\n")
        out.append("```\n" + json.dumps({"decomposition": b["decomposition"], "declaration_checks": b["declarations"]["checks"],
                                          "settings": {k: v for k, v in b["settings"].items() if k != "checker_recompute"}}, indent=1) + "\n```\n")
    if "arm_d" in r:
        dd = r["arm_d"]
        out.append(f"## Arm D: stopping gate at theta {t}, announced (live n = {dd['n']}, unscored {dd['n_unscored']})\n")
        has_b1 = "announced_minus_replay_b1" in dd
        hdr = "| Query | Announced (live) | Replay from B0 | Unannounced live (val2_t2) | Announced minus replay B0 | Announced minus val2_t2 |"
        sep = "|---|---|---|---|---|---|"
        if has_b1:
            hdr += " Replay from B1 | Announced minus replay B1 |"
            sep += "---|---|"
        out.append(hdr + "\n" + sep)
        for k, lab in ROWS:
            row = (f"| {lab} | {pci(dd['observed'][k], k)} | {pci(r['replay_b0'][k], k)} | {pci(r['val2_t2']['observed'][k], k)} | "
                   f"{dci(dd['announced_minus_replay_b0'][k], k)} | {dci(dd['announced_minus_val2_t2'][k], k)} |")
            if has_b1:
                row += f" {pci(r['b1']['replay_b1'][k], k)} | {dci(dd['announced_minus_replay_b1'][k], k)} |"
            out.append(row)
        rm = dd["rmst_flag_free_steps"]
        out.append("\nRestricted mean flag-free steps over 15 steps (Kaplan-Meier): " + json.dumps({k: v for k, v in rm.items() if k != "hazards_descriptive"}) + "\n")
        out.append("Exact harm and abort comparisons: " + json.dumps(dd["exact_harm"]) + "\n")
        out.append("\nPer-step hazard of a first proposal above theta (episode bootstrap):\n")
        out.append("```\n" + json.dumps(dd["first_above_theta_hazard_per_step"], indent=1) + "\n```\n")
        out.append("```\n" + json.dumps(dd["proposal_stats"], indent=1) + "\n```\n")
    if "b1" in r:
        b1 = r["b1"]
        out.append(f"## B1: second baseline, batch-invariant kernels (n = {b1['n']}, unscored {b1['n_unscored']})\n")
        out.append("| Query | B1 [95% CI] | B0 [95% CI] | B1 minus B0 [95% CI] | Replay at theta from B1 | Replay pooled B0+B1 |\n|---|---|---|---|---|---|")
        for k, lab in ROWS:
            if k == "abort_rate":
                continue
            out.append(f"| {lab} | {pci(b1['baseline'][k], k)} | {pci(b1['b0_baseline'][k], k)} | {dci(b1['b1_minus_b0'][k], k)} | "
                       f"{pci(b1['replay_b1'][k], k)} | {pci(b1['replay_pooled'][k], k)} |")
        out.append(f"\nReplay at theta, B1 minus B0 (abort rate): {dci(b1['replay_b1_minus_replay_b0']['abort_rate'], 'abort_rate')}\n")
        pe = b1["primary_p_harm_b1_minus_b0_exact"]
        out.append(f"Primary (B1): P(harm) B1 {pe['k']}/{pe['n']} = {pe['p']:.3f} minus B0 {pe['k_ref']}/{pe['n_ref']} = {pe['p_ref']:.3f}: "
                   f"{pe['diff']:.3f}, Newcombe [{pe['newcombe_95'][0]:.3f}, {pe['newcombe_95'][1]:.3f}].\n")
        if b1.get("replay_b1_vs_val2_live"):
            out.append("### Replay from B1 against the val2 live runs (crosses the kernel switch)\n")
            out.append("| theta | Query | Live val2 | Replay B1 | Live minus replay B1 [95%] | Replay B0 | Live minus replay B0 [95%] |\n|---|---|---|---|---|---|---|")
            for th, v in b1["replay_b1_vs_val2_live"].items():
                for k in ("abort_rate", "p_harm", "p_failure", "mean_total_tokens", "cvar90_total_tokens"):
                    out.append(f"| {th} | {k} | {pci(v['live'][k], k)} | {pci(v['replay_b1'][k], k)} | {dci(v['live_minus_replay_b1'][k], k)} | "
                               f"{pci(v['replay_b0'][k], k)} | {dci(v['live_minus_replay_b0'][k], k)} |")
            out.append("")
        if b1.get("certification"):
            out.append("### Certification on B1 of the thresholds chosen on B0 (exact one-sided 95% upper bound on replayed P(harm))\n")
            out.append("| theta | B1 harmed / n | B1 certifiable level | B0 harmed / n | B0 certifiable level |\n|---|---|---|---|---|")
            for th, v in b1["certification"].items():
                out.append(f"| {th} | {v['b1_replay_harm_k_n'][0]}/{v['b1_replay_harm_k_n'][1]} | {v['b1_certifiable_level_exact_one_sided_95']:.3f} | "
                           f"{v['b0_replay_harm_k_n'][0]}/{v['b0_replay_harm_k_n'][1]} | {v['b0_certifiable_level_exact_one_sided_95']:.3f} |")
            out.append("")
        out.append("```\n" + json.dumps({k: b1[k] for k in ("main_task_success", "limits", "score_stats", "b0_score_stats", "per_step_vs_scorer_tokens_mismatch", "unscored")}, indent=1) + "\n```\n")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
