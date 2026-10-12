# Agent Risk Experiment Harness and Analysis

This directory contains the experiment execution harness library (`harness/agent_risk_harness/`) and the analysis scripts (`analysis/`) for the empirical evaluations reported in the paper.

## Directory Layout

- `harness/`: The `agent_risk_harness` Python package and its `pyproject.toml`.
  - Implements the agent evaluation policies, task wrappers, gate monitors, loss computation, audit logging, and sandbox snapshot/restoration hooks.
- `analysis/`: Analysis, extraction, verification, report, and scripted check scripts.
  - Implements statistical estimation (Clopper-Pearson exact limits, Wilson score intervals, Newcombe paired and unpaired score intervals, cluster and paired bootstraps), identification rules, and paper figure/table regeneration.
- `figures/`: `make_figures.py` regenerates Figures D.1, D.2, and D.3 (files `fig3_forest.pdf`, `fig4_bound.pdf`, `fig_checks.pdf`) from the released analysis outputs and writes `figure_values.json`; Figures 1 and 2 are drawn diagrams, shipped as PDFs.

### Prerequisites

- **Analysis scripts & tests:** Require Python 3.9+ with NumPy (`>=1.20`) and SciPy (`>=1.7`). A3 and A38 also need Matplotlib, and A23 needs Python 3.14 or later for its extraction step (see `ARTIFACT.md`).
- **Execution harness:** Requires Python $\ge 3.11$, ControlArena, and (for E3/E5) $\tau^2$-bench.

## Configuring Log Paths

All analysis scripts locate experiment logs via either:
1. The environment variable `AGENT_RISK_LOGS`:
   ```bash
   export AGENT_RISK_LOGS=/path/to/logs
   ```
   If unset, scripts default to `data/logs` relative to the repository root.
   Scripts resolving paths via `AGENT_RISK_LOGS` include: `analyze_replication.py`, `analyze_d64.py`, `analyze_d65.py`, `describe_d65_incident.py`, `analyze_relay.py`, `analyze_e1.py`, `e2_analyze.py`, `analyze_arms_cd.py`, `analyze_n2.py`, `compute_p1_p2.py`, `test_n2.py`, and all post hoc scripts in `experiments/analysis/posthoc/`.
2. Explicit command-line arguments:
   Other scripts (`analyze_e7.py`, `e7_report_text.py`, `d60_rescore.py`, `disk_watchdog_d66.py`, `e3/*`, `e5/e5_analyze.py`, and `arm-e/arm_e_analyze.py`) take explicit path arguments (e.g., `--logroot data/logs/...`, `--analysis data/logs/...`, `--prediction data/logs/...`).

## Analysis Scripts and Paper Displays

The analysis scripts regenerate the following results and figures reported in the paper:

| Script | Paper Display / Section | Run Codes / Datasets | Primary Outputs |
| :--- | :--- | :--- | :--- |
| `analyze_replication.py` | Section 4.1, Appendix D.3 (Replication) | `D58` (`B2`, `Val3`) | Live stopping gate vs replayed baseline equivalence ($n=1,000$, $\theta=0.94$). |
| `compute_p1_p2.py` | Section 4.2 & Appendix D.3 (Certificates & Ceilings) | `B1`, `B2` (`D58`) | Held-out stopping-gate ceilings ($0.064$ on 12/300; $0.159$ on 37/300); replication ceilings ($0.051$ on 39/1000; $0.111$ on 94/1000); stopping-gate sensitivity and kill-switch replay. |
| `analyze_d64.py` | Section 4.6 (Prompt Intervention / Oversight Cues) | `D64` (`A`, `U`, `B`) | Announced vs unannounced live stopping gate abort rate paired contrast ($n=400$, $\theta=0.94$). |
| `analyze_arms_cd.py` | Section 4.1 (Intervention Arms) | `Arm C`, `Arm D`, `B1`, `B0` | Arm C live retry gate; Arm D announced stopping gate; baseline comparisons; retry bound comparisons. |
| `analyze_n2.py` | Section 4.2 & Appendix D.3 (N2 Estimation) | `N2 stage 1`, `B1`, `Arm C` | Graded retry gate interval from executed continuations; step-0 stratum comparison with live Arm C. |
| `analyze_d65.py` | Section 4.3 & Appendix D.3 (Horizon Scaling) | `D65` (split VMs) | Truncation curves for $k=1..40$ step horizons; baseline harm, replayed stopping harm, retry bounds. |
| `analyze_relay.py` | Section 4.5 (Compositional Relay) | `D66` | Two-shift composition: P1 clean start vs relay, P2 snapshot vs relay, P3 linked vs product dependence. |
| `analyze_e1.py` | Section 4.1 & Table 2 (Single-Gate Targeted Execution) | `E1` | Identified retry gate loss from pass-all snapshots vs paired live retry arm. |
| `e2_analyze.py` | Section 4.1 & Table 2 (Continuous Monitor Calibration) | `E2` | Monitor log-doubt calibration (pilot) and confirmatory stopping gate equivalence. |
| `analyze_e7.py` | Section 4.1 & Table 2 (Two-Gate Targeted Execution) | `E7` | Two-gate identified retry values from pass-all snapshots vs paired live gates ($n=7,200$). |
| `e7_report_text.py` | Section 4.1 & Table 2 (E7 Audit Report) | `E7` | Hash-checked report text for E7 first-block counts, continuation calls, and contrasts. |
| `e3/e3_analyze.py` | Section 4.1 & Table 2 (Airline Demonstration E3) | `E3` | Replayed stopping gate and equivalence analysis for the tau2-bench airline customer-service agent. |
| `e5/e5_analyze.py` | Section 4.1 & Table 2 (Airline Retry Gate E5) | `E5` | Targeted execution evaluation of the airline retry gate from replayed pass-all logs. |


## Running Analysis Scripts

From the repository root:

### Replication (D58)
```bash
python experiments/analysis/analyze_replication.py \
    --manifest data/manifests/d58/manifest_d58.json \
    --logroot data/logs/d58 \
    --b1 data/logs/arms-cd/b1/episodes.jsonl \
    --B 2000 \
    --out outputs/d58/analysis_d58.json
```

### Stopping-Gate Ceilings & Sensitivity
```bash
python experiments/analysis/compute_p1_p2.py
```

### Prompt Intervention (D64)
```bash
python experiments/analysis/analyze_d64.py \
    --manifest data/manifests/d64/manifest_d64.json \
    --logroot data/logs/d64 \
    --B 2000 \
    --out outputs/d64/analysis_d64.json
```

### Compositional Relay (D66)
```bash
python experiments/analysis/analyze_relay.py \
    --manifest data/manifests/d66/manifest_d66.json \
    --logroot data/logs/d66 \
    --B 2000 \
    --out outputs/d66/analysis_d66.json
```

### Two-Gate Targeted Execution (E7)
```bash
python experiments/analysis/analyze_e7.py compare \
    --manifest data/manifests/e7/manifest_e7.json \
    --logroot data/logs/e7 \
    --prediction data/logs/e7/prediction_e7.json \
    --prediction-sha256 a3bccf79de31846adfb037155d4f03017fad537fafabeacad3681fc6086b1bd7 \
    --delta 0.015 \
    --B 4000 \
    --out outputs/e7/analysis_e7.json

python experiments/analysis/e7_report_text.py \
    --analysis data/logs/e7/analysis_e7.json \
    --prediction data/logs/e7/prediction_e7.json \
    --registered-prediction-sha256 a3bccf79de31846adfb037155d4f03017fad537fafabeacad3681fc6086b1bd7 \
    --out outputs/e7/report_text_e7.json
```

## Running Unit & Regression Tests

Offline self-tests and regression test suites can be executed with standard Python:
```bash
cd experiments/analysis
python analyze_replication.py --selftest
python analyze_d64.py --selftest
python analyze_d65.py --selftest
python analyze_relay.py --selftest
python analyze_e1.py selftest
python analyze_e7.py selftest
python e2_analyze.py --selftest
python test_e1.py --groups manifest analysis watchdog
python test_relay.py --groups analysis watchdog probe
python test_e7_report_text.py
```

Tests that require experimental logs (`AGENT_RISK_LOGS`, default `data/logs`):
```bash
cd experiments/analysis
python test_n2.py
```

## Excluded Runs

Runs for other studies are not included.
