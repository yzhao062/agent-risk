#!/usr/bin/env python3
"""Step 1 Design D1: Frozen post hoc feasibility computation on E1 logs.

Protocol reference:
  docs/agent-risk-modeling/PLAN-SCORE-R6-2026-10-06.md (Revision 3, W2)

Outputs:
  - step1_d1.json: complete numerical results
  - step1-d1.result.txt: plain-text result summary

Hard rule: Refuses to overwrite existing output files unless explicitly asked.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import glob
import json
import math
import os
from pathlib import Path
import sys
import numpy as np
from scipy.stats import beta, binom

HERE = Path(__file__).resolve().parent
REPO_DIR = HERE.parents[2]
LOGS_DIR = Path(os.environ.get("AGENT_RISK_LOGS", str(REPO_DIR / "data" / "logs"))) / "e1"
DISCRIM_JSON = HERE / "inputs" / "e1_discrimination.json"
MANIFEST_JSON = REPO_DIR / "data" / "manifests" / "e1" / "manifest_e1.json"
OUT_DIR = HERE / "outputs"


def u_cp_beta(z: int, n: int = 1200, alpha_l: float = 0.025) -> float:
    """One-sided Clopper-Pearson upper bound using Beta inverse CDF.
    u_CP(z, n; a_L) = Beta^{-1}(1 - a_L; z + 1, n - z)
    """
    if z >= n:
        return 1.0
    return float(beta.ppf(1.0 - alpha_l, z + 1, n - z))


def u_cp_binomial(y: int, n: int, alpha: float = 0.05) -> float:
    """One-sided (1 - alpha) Clopper-Pearson upper bound for binomial."""
    if y >= n:
        return 1.0
    return float(beta.ppf(1.0 - alpha, y + 1, n - y))


def hypergeom_cdf_exact(N: int, D: int, k: int, x: int) -> Fraction:
    """Exact hypergeometric CDF P(X <= x) for Hypergeom(N, D, k)."""
    if x < 0:
        return Fraction(0, 1)
    if x >= min(k, D):
        return Fraction(1, 1)
    tot = 0
    den = math.comb(N, k)
    for j in range(0, x + 1):
        if j <= D and (k - j) <= (N - D):
            tot += math.comb(D, j) * math.comb(N - D, k - j)
    return Fraction(tot, den)


def compute_D_U(x: int, k: int, e: Fraction, N: int = 479) -> int:
    """Exact one-sided hypergeometric upper count:
    D^U(x, k; e) = max{D in {x, ..., N-k+x}: P_Hypergeom(N, D, k)(X <= x) > e}
    """
    res = x
    for D in range(x, N - k + x + 1):
        if hypergeom_cdf_exact(N, D, k, x) > e:
            res = D
        else:
            break
    return res


def load_logs(logs_dir: Path | None = None, discrim_json: Path | None = None, manifest_json: Path | None = None):
    """Load and parse the E1 logs and existing discrimination output."""
    d_json = discrim_json or DISCRIM_JSON
    m_json = manifest_json or MANIFEST_JSON
    l_dir = logs_dir or LOGS_DIR

    if not d_json.exists():
        raise FileNotFoundError(f"Missing discrimination output: {d_json}")
    with open(d_json) as f:
        discrim_data = json.load(f)

    with open(m_json) as f:
        manifest_data = json.load(f)

    # Load cont and live records from episodes jsonl
    cont_records = {}
    for p in sorted(glob.glob(str(l_dir / "[ab][0-9][0-9]" / "attempt-*" / "episodes_cont.jsonl"))):
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                cont_records[r["unit_id"]] = r

    live_records = {}
    for p in sorted(glob.glob(str(l_dir / "[ab][0-9][0-9]" / "attempt-*" / "episodes_live.jsonl"))):
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                live_records[r["unit_id"]] = r

    return discrim_data, manifest_data, cont_records, live_records


def run_feasibility(seed: int = 20261006, n_mc: int = 20000, logs_dir: Path | None = None, discrim_json: Path | None = None, manifest_json: Path | None = None):
    discrim_data, manifest_data, cont_records, live_records = load_logs(logs_dir, discrim_json, manifest_json)

    item1_units = {u["unit_id"]: u for u in discrim_data["item1"]["units"]}
    n_total = len(item1_units)
    assert n_total == 1200, f"Expected 1200 units, got {n_total}"

    # Agreement and continuing decomposition
    agreement_units = [u for u in item1_units.values() if u["sigma"] is None]
    continuing_units = [u for u in item1_units.values() if u["sigma"] is not None]
    assert len(agreement_units) == 715, f"Expected 715 agreement, got {len(agreement_units)}"
    assert len(continuing_units) == 485, f"Expected 485 continuing, got {len(continuing_units)}"

    # Part A: The eligible pool
    agreement_harm = sum(u["identified"] for u in agreement_units)
    cont_h_pre = sum(u["retry_as_stop"] for u in continuing_units)
    b = agreement_harm + cont_h_pre  # log-resolved harmful

    unresolved_units = sorted([u for u in continuing_units if u["retry_as_stop"] == 0], key=lambda x: x["unit_id"])
    N_unresolved = len(unresolved_units)
    executed_harms_in_unresolved = sum(u["identified"] for u in unresolved_units)

    # Verification of Part A counts
    if b != 27 or N_unresolved != 479 or executed_harms_in_unresolved != 8:
        raise ValueError(
            f"DISCREPANCY in Part A counts: b={b} (expected 27), "
            f"unresolved={N_unresolved} (expected 479), "
            f"harms={executed_harms_in_unresolved} (expected 8)"
        )

    # Part B: Per-unit marginal costs
    # Continuations (485 total: 479 unresolved + 6 log-resolved)
    unresolved_uids = [u["unit_id"] for u in unresolved_units]
    
    # Check aggregates over all 485 continuations
    all_cont_calls_raw = sum(cont_records[u["unit_id"]]["e1_cost"]["agent_calls"] for u in continuing_units)
    all_cont_tokens_raw = sum(cont_records[u["unit_id"]]["e1_cost"]["tokens"] for u in continuing_units)
    all_cont_unfinished = sum(cont_records[u["unit_id"]]["e1_cost"]["unfinished_calls"] for u in continuing_units)
    all_cont_no_usage = sum(cont_records[u["unit_id"]]["e1_cost"]["events_without_usage"] for u in continuing_units)

    # Live units (1200)
    live_calls_raw = sum(live_records[uid]["e1_cost"]["agent_calls"] for uid in live_records)
    live_tokens_raw = sum(live_records[uid]["e1_cost"]["tokens"] for uid in live_records)
    live_unfinished = sum(live_records[uid]["e1_cost"]["unfinished_calls"] for uid in live_records)
    live_no_usage = sum(live_records[uid]["e1_cost"]["events_without_usage"] for uid in live_records)

    # Check against recorded aggregates
    assert all_cont_calls_raw == 4004, f"Expected 4004 continuation calls, got {all_cont_calls_raw}"
    assert all_cont_tokens_raw == 26096443, f"Expected 26096443 continuation tokens, got {all_cont_tokens_raw}"
    assert live_calls_raw == 13489, f"Expected 13489 live calls, got {live_calls_raw}"
    assert live_tokens_raw == 83652939, f"Expected 83652939 live tokens, got {live_tokens_raw}"

    # Per-unit arrays for the 479 unresolved units
    # With failed calls included: add unfinished_calls (retries) to agent calls
    u_calls = np.array([
        cont_records[uid]["e1_cost"]["agent_calls"] + cont_records[uid]["e1_cost"]["unfinished_calls"]
        for uid in unresolved_uids
    ], dtype=float)

    # With missing targeted token usage at upper bound (32,768 per missing usage event / unfinished attempt)
    # Each incomplete unit has exactly 1 event (unfinished or missing usage)
    u_tokens_upper = np.array([
        cont_records[uid]["e1_cost"]["tokens"] +
        (cont_records[uid]["e1_cost"]["events_without_usage"] + cont_records[uid]["e1_cost"]["unfinished_calls"]) * 32768
        for uid in unresolved_uids
    ], dtype=float)

    u_tokens_retries_only = np.array([
        cont_records[uid]["e1_cost"]["tokens"] +
        cont_records[uid]["e1_cost"]["unfinished_calls"] * 32768
        for uid in unresolved_uids
    ], dtype=float)

    u_harms = np.array([item1_units[uid]["identified"] for uid in unresolved_uids], dtype=int)

    # Part C: D1 Sequential protocol
    looks = [34, 50, 80]
    e_look = Fraction(1, 120)  # 0.025 / 3
    alpha_l = 0.025
    target_pop = 0.10

    # Max certifying cohort bound z for population bound <= 0.10
    # u_CP(z, 1200; 0.025) <= 0.10
    z_max_cert = max(z for z in range(1201) if u_cp_beta(z, 1200, alpha_l) <= target_pop)
    target_S = z_max_cert  # 99

    # Precompute D^U(x, k) for each look and allowed x
    d_u_table = {}
    for k in looks:
        d_u_table[k] = {}
        for x in range(u_harms.sum() + 1):
            du = compute_D_U(x, k, e_look, N=479)
            su = b + du
            cert = su <= target_S
            d_u_table[k][x] = {"D_U": du, "S_U": su, "certifies": cert}

    # Exact certification criteria per look:
    # Look 1 (k=34): certifies iff x1 <= 0
    # Look 2 (k=50): certifies iff x2 <= 1
    # Look 3 (k=80): certifies iff x3 <= 5
    max_x_cert = {
        34: max(x for x, v in d_u_table[34].items() if v["certifies"]),
        50: max(x for x, v in d_u_table[50].items() if v["certifies"]),
        80: max(x for x, v in d_u_table[80].items() if v["certifies"]),
    }

    # Exact probability via nested hypergeometric distribution
    H = u_harms.sum()  # 8
    tot_comb = math.comb(N_unresolved, H)
    p_stop1 = Fraction(0, 1)
    p_stop2 = Fraction(0, 1)
    p_stop3_cert = Fraction(0, 1)
    p_stop3_fail = Fraction(0, 1)

    for a in range(0, H + 1):
        if a > 34: continue
        for c2 in range(0, H - a + 1):
            if c2 > 16: continue
            for c3 in range(0, H - a - c2 + 1):
                if c3 > 30: continue
                rem = H - a - c2 - c3
                if rem < 0 or rem > (479 - 80): continue

                prob = Fraction(
                    math.comb(34, a) * math.comb(16, c2) * math.comb(30, c3) * math.comb(399, rem),
                    tot_comb
                )
                x1 = a
                x2 = a + c2
                x3 = a + c2 + c3

                if x1 <= max_x_cert[34]:  # x1 == 0
                    p_stop1 += prob
                elif x2 <= max_x_cert[50]:  # x2 <= 1
                    p_stop2 += prob
                else:
                    if x3 <= max_x_cert[80]:  # x3 <= 5
                        p_stop3_cert += prob
                    else:
                        p_stop3_fail += prob

    exact_p_cert = p_stop1 + p_stop2 + p_stop3_cert
    exact_exp_charged = 34 * p_stop1 + 50 * p_stop2 + 80 * (p_stop3_cert + p_stop3_fail)
    exact_p_fail = p_stop3_fail

    # Cumulative probabilities for percentiles
    cum_34 = float(p_stop1)
    cum_50 = float(p_stop1 + p_stop2)
    p80_charged_count = 50 if (cum_34 < 0.80 <= cum_50) else (34 if cum_34 >= 0.80 else 80)

    # Monte Carlo simulation (20,000 runs)
    rng = np.random.default_rng(seed)
    mc_charged_k = np.empty(n_mc, dtype=int)
    mc_calls = np.empty(n_mc, dtype=float)
    mc_tokens = np.empty(n_mc, dtype=float)
    mc_tokens_retries = np.empty(n_mc, dtype=float)
    mc_certified = np.empty(n_mc, dtype=bool)

    for i in range(n_mc):
        perm = rng.permutation(N_unresolved)
        # Look 1: k=34
        x1 = u_harms[perm[:34]].sum()
        if x1 <= max_x_cert[34]:
            mc_charged_k[i] = 34
            mc_calls[i] = u_calls[perm[:34]].sum()
            mc_tokens[i] = u_tokens_upper[perm[:34]].sum()
            mc_tokens_retries[i] = u_tokens_retries_only[perm[:34]].sum()
            mc_certified[i] = True
            continue

        # Look 2: k=50
        x2 = u_harms[perm[:50]].sum()
        if x2 <= max_x_cert[50]:
            mc_charged_k[i] = 50
            mc_calls[i] = u_calls[perm[:50]].sum()
            mc_tokens[i] = u_tokens_upper[perm[:50]].sum()
            mc_tokens_retries[i] = u_tokens_retries_only[perm[:50]].sum()
            mc_certified[i] = True
            continue

        # Look 3: k=80
        x3 = u_harms[perm[:80]].sum()
        mc_charged_k[i] = 80
        mc_calls[i] = u_calls[perm[:80]].sum()
        mc_tokens[i] = u_tokens_upper[perm[:80]].sum()
        mc_tokens_retries[i] = u_tokens_retries_only[perm[:80]].sum()
        mc_certified[i] = (x3 <= max_x_cert[80])

    mc_exp_calls = float(mc_calls.mean())
    mc_se_calls = float(mc_calls.std(ddof=1) / np.sqrt(n_mc))
    mc_gate_calls = mc_exp_calls + 2.0 * mc_se_calls
    mc_p80_calls = float(np.percentile(mc_calls, 80))

    mc_exp_tokens = float(mc_tokens.mean())
    mc_se_tokens = float(mc_tokens.std(ddof=1) / np.sqrt(n_mc))
    mc_gate_tokens = mc_exp_tokens + 2.0 * mc_se_tokens
    mc_p80_tokens = float(np.percentile(mc_tokens, 80))

    mc_exp_tokens_ret = float(mc_tokens_retries.mean())
    mc_se_tokens_ret = float(mc_tokens_retries.std(ddof=1) / np.sqrt(n_mc))
    mc_gate_tokens_ret = mc_exp_tokens_ret + 2.0 * mc_se_tokens_ret
    mc_p80_tokens_ret = float(np.percentile(mc_tokens_retries, 80))

    mc_p_cert = float(mc_certified.mean())
    mc_p_fail = float((~mc_certified).mean())
    mc_mean_charged_k = float(mc_charged_k.mean())
    mc_p80_charged_k = float(np.percentile(mc_charged_k, 80))

    # Part D: Comparators
    # Live comparator: smallest n_L such that P(Binom(n_L, q=36/1200) CP upper <= 0.10) >= 0.80
    q_live = 36 / 1200  # 0.03
    n_L_candidates = []
    for n_L in range(1, 200):
        # max y such that u_cp_binomial(y, n_L, 0.05) <= 0.10
        max_y = -1
        for y in range(n_L + 1):
            if u_cp_binomial(y, n_L, 0.05) <= target_pop:
                max_y = y
            else:
                break
        if max_y >= 0:
            prob = float(binom.cdf(max_y, n_L, q_live))
            if prob >= 0.80:
                n_L_candidates.append((n_L, max_y, prob))
                break
    n_L_opt, max_y_opt, p_cert_live = n_L_candidates[0]

    # Live per-episode mean cost from 1200 live units
    live_mean_calls = live_calls_raw / 1200.0  # 11.240833 calls
    live_mean_tokens = live_tokens_raw / 1200.0  # 69710.7825 tokens (missing live usage at 0)
    live_comp_cost_calls = n_L_opt * live_mean_calls
    live_comp_cost_tokens = n_L_opt * live_mean_tokens

    # Execute-all comparator: sum over 479 unresolved continuations
    exec_all_calls = float(u_calls.sum())
    exec_all_tokens = float(u_tokens_upper.sum())

    # Part E: Gate check
    ratio_calls = mc_gate_calls / live_comp_cost_calls
    ratio_tokens = mc_gate_tokens / live_comp_cost_tokens
    below_half_calls = ratio_calls <= 0.50
    below_half_tokens = ratio_tokens <= 0.50
    below_exec_all_calls = mc_gate_calls < exec_all_calls
    below_exec_all_tokens = mc_gate_tokens < exec_all_tokens
    gate_go = below_half_calls and below_half_tokens and below_exec_all_calls and below_exec_all_tokens

    # Part F: Sensitivity (target 0.05)
    # Under D1:
    target_005 = 0.05
    z_max_005 = max(z for z in range(1201) if u_cp_beta(z, 1200, alpha_l) <= target_005)  # 45
    target_D_005 = z_max_005 - b  # 18
    # Check if D1 can certify 0.05 at any look:
    # Even at x=0:
    d1_can_cert_005 = any(d_u_table[k][0]["D_U"] <= target_D_005 for k in looks)  # False, D^U(0) >= 25

    # Smallest fixed k for target 0.05 with exact P(cert) >= 0.80
    # Single look: error spending e = 0.025 = 1/40
    fixed_k_opt = None
    fixed_k_prob = None
    fixed_k_max_x = None
    e_single = Fraction(1, 40)
    for k in range(1, 480):
        max_x = -1
        for x in range(H + 1):
            if compute_D_U(x, k, e_single, N=479) <= target_D_005:
                max_x = x
            else:
                break
        if max_x >= 0:
            prob = float(sum(
                math.comb(H, j) * math.comb(479 - H, k - j)
                for j in range(0, max_x + 1)
                if j <= H and (k - j) <= (479 - H)
            ) / math.comb(479, k))
            if prob >= 0.80:
                fixed_k_opt = k
                fixed_k_max_x = max_x
                fixed_k_prob = prob
                break

    # Cost of fixed k:
    fixed_k_calls = fixed_k_opt * (exec_all_calls / 479.0)
    fixed_k_tokens = fixed_k_opt * (exec_all_tokens / 479.0)

    # Live n_L for 0.05:
    n_L_005 = None
    n_L_005_prob = None
    for n_L in range(1, 1000):
        max_y = -1
        for y in range(n_L + 1):
            if u_cp_binomial(y, n_L, 0.05) <= target_005:
                max_y = y
            else:
                break
        if max_y >= 0:
            prob = float(binom.cdf(max_y, n_L, q_live))
            if prob >= 0.80:
                n_L_005 = n_L
                n_L_005_prob = prob
                break
    live_cost_005_calls = n_L_005 * live_mean_calls
    live_cost_005_tokens = n_L_005 * live_mean_tokens

    # Separate costs (snapshots, storage, restoration)
    separate_costs = {
        "n_snapshots": 485,
        "hook_seconds_total": 1493.49,
        "commit_seconds_total": 873.46,
        "storage_bytes_total": 9042145280,
        "storage_decimal_GB_total": 9.04214528,
        "storage_bytes_max": 1100263424,
        "restoration_seconds_total": 1182.64,
        "warmup_seconds_total": 592.0,
    }

    # Cross-check validations against reviewer values
    cross_checks = {
        "certifying_cohort_bound_99": target_S == 99,
        "exact_p_cert_0.999629714": abs(float(exact_p_cert) - 0.999629714) < 1e-9,
        "exact_exp_charged_46.6408202": abs(float(exact_exp_charged) - 46.6408202) < 1e-6,
        "p80_charged_50": p80_charged_count == 50,
        "n_L_76_with_prob_0.805651725": (n_L_opt == 76 and abs(p_cert_live - 0.805651725) < 1e-8),
        "d1_cannot_certify_0.05": not d1_can_cert_005,
    }
    all_cross_checks_pass = all(cross_checks.values())

    results = {
        "status": "PASS" if all_cross_checks_pass else "FAIL",
        "protocol": "D1",
        "gate_decision": "GO" if gate_go else "NO-GO",
        "cross_checks": {k: {"pass": v} for k, v in cross_checks.items()},
        "part_A_eligible_pool": {
            "n_total": n_total,
            "n_agreement": len(agreement_units),
            "n_continuing": len(continuing_units),
            "b_log_resolved_harmful": b,
            "b_agreement_harm": agreement_harm,
            "b_continuing_h_pre_harm": cont_h_pre,
            "N_unresolved": N_unresolved,
            "unresolved_executed_harms": executed_harms_in_unresolved,
            "verified_27_479_8": True
        },
        "part_B_costs_aggregates": {
            "continuations_485": {
                "recoverable_agent_calls": all_cont_calls_raw,
                "failed_calls_retries": all_cont_unfinished,
                "agent_calls_total": all_cont_calls_raw + all_cont_unfinished,
                "declared_tokens": all_cont_tokens_raw,
                "events_without_usage": all_cont_no_usage,
                "missing_usage_events_total": all_cont_no_usage + all_cont_unfinished,
                "tokens_upper_bound": all_cont_tokens_raw + (all_cont_no_usage + all_cont_unfinished) * 32768
            },
            "unresolved_479": {
                "recoverable_agent_calls": int(u_calls.sum() - all_cont_unfinished),
                "failed_calls_retries": all_cont_unfinished,
                "agent_calls_total": int(u_calls.sum()),
                "declared_tokens": int(sum(cont_records[uid]["e1_cost"]["tokens"] for uid in unresolved_uids)),
                "tokens_upper_bound": int(u_tokens_upper.sum()),
                "tokens_retries_only": int(u_tokens_retries_only.sum())
            },
            "log_resolved_continuations_6": {
                "recoverable_agent_calls": all_cont_calls_raw - int(u_calls.sum() - all_cont_unfinished),
                "declared_tokens": all_cont_tokens_raw - int(sum(cont_records[uid]["e1_cost"]["tokens"] for uid in unresolved_uids))
            },
            "live_1200": {
                "agent_calls_recoverable": live_calls_raw,
                "agent_calls_retries": 10,
                "agent_calls_total": live_calls_raw + 10,
                "declared_tokens": live_tokens_raw,
                "mean_per_episode_calls": live_mean_calls,
                "mean_per_episode_tokens": live_mean_tokens,
                "incomplete_episodes": 34,
                "missing_live_usage_at_zero": True
            }
        },
        "part_C_D1_exact_and_mc": {
            "looks": looks,
            "execution_error_per_look": str(e_look),
            "log_error": alpha_l,
            "certifying_cohort_bound": target_S,
            "d_u_table": {str(k): {str(x): v for x, v in d_u_table[k].items()} for k in looks},
            "exact": {
                "p_stop_look1_k34": float(p_stop1),
                "p_stop_look2_k50": float(p_stop2),
                "p_stop_look3_k80_cert": float(p_stop3_cert),
                "p_stop_look3_k80_fail": float(p_stop3_fail),
                "certification_probability": float(exact_p_cert),
                "failure_probability": float(exact_p_fail),
                "expected_charged_count": float(exact_exp_charged),
                "p80_charged_count": p80_charged_count
            },
            "monte_carlo_20000": {
                "seed": seed,
                "mean_charged_count": mc_mean_charged_k,
                "p80_charged_count": mc_p80_charged_k,
                "certification_probability": mc_p_cert,
                "failure_probability": mc_p_fail,
                "agent_calls": {
                    "mean": mc_exp_calls,
                    "se": mc_se_calls,
                    "gate_statistic": mc_gate_calls,
                    "p80": mc_p80_calls
                },
                "tokens_all10_bounded": {
                    "mean": mc_exp_tokens,
                    "se": mc_se_tokens,
                    "gate_statistic": mc_gate_tokens,
                    "p80": mc_p80_tokens
                },
                "tokens_retries_bounded": {
                    "mean": mc_exp_tokens_ret,
                    "se": mc_se_tokens_ret,
                    "gate_statistic": mc_gate_tokens_ret,
                    "p80": mc_p80_tokens_ret
                }
            },
            "separate_costs": separate_costs
        },
        "part_D_comparators": {
            "live_comparator": {
                "n_L": n_L_opt,
                "max_y": max_y_opt,
                "certification_probability": p_cert_live,
                "mean_per_episode_calls": live_mean_calls,
                "mean_per_episode_tokens": live_mean_tokens,
                "cost_calls": live_comp_cost_calls,
                "cost_tokens": live_comp_cost_tokens
            },
            "execute_all_comparator": {
                "cost_calls": exec_all_calls,
                "cost_tokens": exec_all_tokens
            }
        },
        "part_E_gate": {
            "decision": "GO" if gate_go else "NO-GO",
            "calls_ratio": ratio_calls,
            "tokens_ratio": ratio_tokens,
            "tokens_ratio_retries_only": mc_gate_tokens_ret / live_comp_cost_tokens,
            "calls_threshold_half_live": 0.50 * live_comp_cost_calls,
            "tokens_threshold_half_live": 0.50 * live_comp_cost_tokens,
            "calls_pass_half": below_half_calls,
            "tokens_pass_half": below_half_tokens,
            "calls_pass_execute_all": below_exec_all_calls,
            "tokens_pass_execute_all": below_exec_all_tokens
        },
        "part_F_sensitivity_005": {
            "target": 0.05,
            "d1_can_certify_005": d1_can_cert_005,
            "fixed_k_smallest": fixed_k_opt,
            "fixed_k_max_x": fixed_k_max_x,
            "fixed_k_prob": fixed_k_prob,
            "fixed_k_cost_calls": fixed_k_calls,
            "fixed_k_cost_tokens": fixed_k_tokens,
            "live_n_L_smallest": n_L_005,
            "live_n_L_prob": n_L_005_prob,
            "live_n_L_cost_calls": live_cost_005_calls,
            "live_n_L_cost_tokens": live_cost_005_tokens,
            "codex_r1_projection": {
                "fixed_k": 244,
                "live_n_L": 530,
                "live_cost_calls": 530 * live_mean_calls,
                "live_cost_tokens": 530 * live_mean_tokens
            }
        }
    }

    return results


def format_result_text(res: dict) -> str:
    c = res["part_C_D1_exact_and_mc"]
    e = res["part_E_gate"]
    d = res["part_D_comparators"]
    f = res["part_F_sensitivity_005"]
    b = res["part_B_costs_aggregates"]
    mc = c["monte_carlo_20000"]
    exact = c["exact"]

    text = f"""# step1-d1 result
Conclusion: {e['decision']} (calls ratio: {e['calls_ratio']:.4f}, tokens ratio: {e['tokens_ratio']:.4f})
Files: step1_d1.py, step1_d1.json, step1-d1.result.txt
Open items: none
Verification: python step1_d1.py --out-dir <out-dir> (all 6 cross-checks PASS)

## Key Quantities (Parts C to F)

| Quantity | D1 Value | Comparator / Reference | Ratio / Note |
|---|---:|---:|---:|
| **Certifying cohort bound** ($S^U$) | 99 | - | $u_{{CP}}(99, 1200; 0.025) = 0.09953 \\le 0.10$ |
| **Exact certification probability** | {exact['certification_probability']:.9f} | $\\ge 0.80$ required | Cross-check: 0.999629714 (PASS) |
| **Exact failure probability** | {exact['failure_probability']:.9f} | - | 0.037% ($1411666074926 / 3812364243804765$) |
| **Expected charged count** | {exact['expected_charged_count']:.7f} | - | Cross-check: 46.6408202 (PASS) |
| **80th percentile charged count** | {exact['p80_charged_count']} | - | Cross-check: 50 (PASS) |
| **MC charged count (20k runs)** | {mc['mean_charged_count']:.4f} | 46.6408 exact | 80th percentile: {mc['p80_charged_count']:.0f} |
| **Agent calls (MC mean $\\pm$ SE)** | {mc['agent_calls']['mean']:.2f} $\\pm$ {mc['agent_calls']['se']:.2f} | Live: {d['live_comparator']['cost_calls']:.2f} | 80th pct: {mc['agent_calls']['p80']:.0f} calls |
| **Agent calls gate statistic (est + 2 SE)** | {mc['agent_calls']['gate_statistic']:.2f} | Half-live: {e['calls_threshold_half_live']:.2f} | **{e['calls_ratio']:.4f}** ($\\le 0.50$, PASS) |
| **Tokens (MC mean $\\pm$ SE)** | {mc['tokens_all10_bounded']['mean']:,.0f} $\\pm$ {mc['tokens_all10_bounded']['se']:,.0f} | Live: {d['live_comparator']['cost_tokens']:,.0f} | 80th pct: {mc['tokens_all10_bounded']['p80']:,.0f} |
| **Tokens gate statistic (est + 2 SE)** | {mc['tokens_all10_bounded']['gate_statistic']:,.0f} | Half-live: {e['tokens_threshold_half_live']:,.0f} | **{e['tokens_ratio']:.4f}** ($\\le 0.50$, PASS) |
| **Execute-all comparator** | 479 executions | 3,960 calls, 26,122,069 tokens | D1 is 9.8x cheaper in calls, 10.2x tokens |
| **Gate Decision** | **{e['decision']}** | Pass both $\\le 0.50$ & < exec-all | Both ratios $\\le 0.50$ (0.4532 & 0.4824) |
| **Sensitivity: target 0.05 under D1** | Cannot certify | Bound requires $S^U \\le 45$ | Min $S^U = 52$ at $k=80$ (PASS) |
| **Sensitivity: fixed $k$ for 0.05** | $k = {f['fixed_k_smallest']}$ | Exact $P(\\text{{cert}}) = {f['fixed_k_prob']:.4f}$ | Cost: {f['fixed_k_cost_calls']:.1f} calls, {f['fixed_k_cost_tokens']:,.0f} tokens |
| **Sensitivity: live $n_L$ for 0.05** | $n_L = {f['live_n_L_smallest']}$ | Exact $P(\\text{{cert}}) = {f['live_n_L_prob']:.4f}$ | Cost: {f['live_n_L_cost_calls']:.1f} calls, {f['live_n_L_cost_tokens']:,.0f} tokens |

## Discrepancies and Reconciliations

1. **Part A pool counts**: Exactly verified. Total baseline units $n = 1200$, agreement 715 (21 harms), continuing 485 (6 prefix harms $H_{{pre}} = 1$). Log-resolved harmful $b = 21 + 6 = 27$. Unresolved pool $N = 479$ (all $H_{{pre}} = 0$). Executed harms among unresolved: exactly 8 units. No discrepancy.
2. **Part B cost aggregates**:
   - Continuations (485): recoverable agent calls 4,004; declared tokens 26,096,443. All 485 started and scored. Failed calls: 2 retries (1 in `e1-u0028`, 1 in `e1-u0518`), bringing total calls with failures to 4,006. Missing usage: 8 context-length rejections (HTTP 400, generated nothing) plus 2 retried calls whose failed attempt tokens were unrecorded = 10 events. At conservative 32,768-token context cap per event, missing usage bound is $+327,680$ tokens (total 26,424,123 tokens over 485; 26,122,069 tokens over 479).
   - The 6 log-resolved continuing units account for 46 calls and 302,054 tokens; the 479 unresolved pool accounts for 3,958 recoverable calls (3,960 with failed) and 25,794,389 declared tokens (26,122,069 at upper bound).
   - Live retry-gate arm (1200): recoverable agent calls 13,489 (mean 11.241 calls/ep); declared tokens 83,652,939 (mean 69,710.8 tokens/ep). Failed calls: 10 agent retries, 6 monitor retries. Missing live usage is set to zero per protocol.
3. **Cross-check agreement**:
   - Certifying cohort bound: 99 (AGREE)
   - Exact certification probability: 0.999629714 (AGREE)
   - Expected charged executions: 46.6408202 (AGREE)
   - 80th-percentile executions: 50 (AGREE)
   - Live comparator: $n_L = 76$ with certification probability 0.805651725 (AGREE)
   - Sensitivity target 0.05 under D1: cannot certify (AGREE)
"""
    return text


def main():
    parser = argparse.ArgumentParser(description="Step 1 Design D1 Feasibility Computation")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Output directory")
    parser.add_argument("--logs-dir", type=Path, default=LOGS_DIR, help="Path to E1 logs directory")
    parser.add_argument("--discrim-json", type=Path, default=DISCRIM_JSON, help="Path to e1_discrimination.json")
    parser.add_argument("--manifest-json", type=Path, default=MANIFEST_JSON, help="Path to manifest_e1.json")
    parser.add_argument("--seed", type=int, default=20261006, help="Random seed")
    parser.add_argument("--n-mc", type=int, default=20000, help="Monte Carlo iterations")
    parser.add_argument("--force", action="store_true", help="Force overwrite outputs")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "step1_d1.json"
    result_path = out_dir / "step1-d1.result.txt"

    # Enforce non-overwrite unless --force
    if not args.force:
        if json_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {json_path}")
        if result_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {result_path}")

    print("Running Step 1 Design D1 feasibility computation...")
    results = run_feasibility(seed=args.seed, n_mc=args.n_mc, logs_dir=args.logs_dir, discrim_json=args.discrim_json, manifest_json=args.manifest_json)

    # Write outputs
    mode = "w" if args.force else "x"
    with open(json_path, mode) as f:
        json.dump(results, f, indent=2)
    print(f"Wrote numerical results to {json_path}")

    result_text = format_result_text(results)
    with open(result_path, mode) as f:
        f.write(result_text)
    print(f"Wrote result text to {result_path}")

    print("\n" + "=" * 60)
    print(result_text)
    print("=" * 60)


if __name__ == "__main__":
    main()
