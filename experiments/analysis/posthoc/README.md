# Post Hoc Analysis Scripts

These scripts compute the paper's post hoc numbers, the analyses that no registration fixed before its run. `ARTIFACT.md` gives the exact command for each (A11, A23 to A25, A28, A32 to A39) and the paper display it supports; this file lists what each script reads and writes. Run the commands from the repository root after downloading the logs to `data/` (see `data/README.md`); the scripts read the logs from `data/logs/` or from the folder named by `AGENT_RISK_LOGS`. Inputs are opened read-only. Several scripts refuse to overwrite an existing output, so move an earlier output away before a rerun. Outputs written next to the scripts are ignored by git.

| Script | ARTIFACT | What it computes | Reads | Writes |
|---|---|---|---|---|
| `analyze_d65_subsets.py` | A11 | Horizon curves of the long-horizon log by host and subset | `data/manifests/d65/`, `data/logs/d65/` | the `--out` file |
| `e1_discrimination.py` | A23 | E1 reference predictors, reproduction after relabelling tool-call identifiers, and snapshot-rule counts | `data/logs/e1/` (including the two serving records), `data/manifests/e1/`, `registrations/E1-PREREG-2026-10-05.md` | the `--out` and `--record` files; an extraction cache under `outputs/` here |
| `step1_d1.py` | A24 | Design D1 of the budget design table from per-unit raw events | `data/logs/e1/`, `data/manifests/e1/`, `inputs/e1_discrimination.json` | `outputs/step1_d1.json` and `outputs/step1-d1.result.txt` here |
| `relay_pathwise.py` | A25 | Pathwise reproduction for the two-agent relay, second agent rerun from the relay's own handoff snapshot | `data/manifests/d66/manifest_d66.json`, `data/logs/d66/` | `relay_pathwise.json` here |
| `logs_final_recount.py` | A28 | Recount of the log-derived numbers in the last paragraph of Appendix D.3 | `data/logs/e1/`, `data/logs/d58/`, `data/logs/e2/` | standard output |
| `snapshot_cost_strata.py` | A28 | Per-step cost strata of the E1 snapshots and the storage and commit-time projections of Appendix D.3 | `data/logs/e1/*/attempt-*/units.json` | standard output |
| `e7_records_posthoc.py` | A32 | Snapshot-design count (Proposition F.10.1) on the E7 pass-all log | `data/manifests/e7/`, `data/logs/e7/` | the `--out` file |
| `ve7.py` | A33 | Independent verification of the E7 run without importing the harness | `data/manifests/e7/manifest_e7.json`, `data/logs/e7/` | `ve7-*` files here |
| `e7_skip_reference.py` | A34 | The "skip the blocked calls" reference, as fixed by `inputs/E7-SKIP-REFERENCE-PREANALYSIS-2026-10-07.md` before it ran | `data/manifests/e7/`, `data/logs/e7/` | the `--out` file |
| `vskip.py` | A34 | Independent recomputation of A34 | `data/logs/e7/`, `inputs/E7-SKIP-REFERENCE-PREANALYSIS.frozen.md` | `vskip-*` files here |
| `e7_joint_snapshots.py` | A35 | Snapshot counts when the query reads harm and task success together | `data/manifests/e7/`, `data/logs/e7/` (loads `e7_records_posthoc.py`) | the `--out` file |
| `posthoc_strata_e7.py` | A36 | Execution bounds across the registered E7 strata (the E7 Check) | `data/logs/e7/prediction_e7.json`, `data/logs/e7/analysis_e7.json` | the `--out` file |
| `verify_e7.py` | A37 | E7 decisions at single and joint 95% coverage, the decomposition, and the agent-call ratios | `data/logs/e7/{prediction_e7,analysis_e7,posthoc_skip_reference_e7}.json`, `experiments/figures/figure_values.json` | `verification_e7.json` here |
| `decision_map_e7.py` | A38 | The declared decision rule over harm ceilings 0.10 to 0.45 | `verification_e7.json` (A37), `inputs/vskip-units.jsonl`, `data/manifests/e7/`, `data/logs/e7/` | `decision_*` files, `paired_units.jsonl`, and manifests here |
| `extract_inputs.py` | A39 | Paired E7 inputs for the existing-log cost analysis | `inputs/paired_units.{csv,jsonl}`, `inputs/audit_e7.py`, `inputs/frozen_harness/`, `source_hashes.json`, `data/logs/e7/` | `paired_data.{csv,npz}`, `metadata.json`, `input_manifest.json` here |
| `analyze_c10.py` | A39 | Live-only subsampling against the continuation cost | the outputs of `extract_inputs.py` | `results.json` and `live_draw_orders.npz` (about 6 MB; it stays after the run) here |
| `supplement_checks.py` | A39 | Checks of the numbers that the supplement quotes from the cost analysis | the outputs of the two scripts above | `supplement_checks.json` and `output_manifest.json` here |

## Inputs

- `inputs/decision_e2.json`: the stronger-agent decision record of A26.
- `inputs/e1_discrimination.json`: the archived output of A23, read by A24.
- `inputs/E7-SKIP-REFERENCE-PREANALYSIS-2026-10-07.md` and its frozen copy `inputs/E7-SKIP-REFERENCE-PREANALYSIS.frozen.md`: the note that fixed the A34 reading before it ran (sha256 `8e7f1bbf6dbf8fafc77ba43b28cb1564b66ef46573b05ba6d4f208f3f1cdf964`).
- `inputs/budget_d1_record.md`: the record of the budget design D1 (A24).
- `inputs/paired_units.{csv,jsonl}`, `inputs/vskip-units.jsonl`, `inputs/audit_e7.py`, and `inputs/frozen_harness/`: the paired E7 extractions and the frozen copies of the harness code that A38 and A39 read.
- `source_hashes.json` (and its copy in `inputs/`): the pinned sha256 of every read-only input of A39; the scripts stop on any mismatch.

One pinned file differs from the original: `inputs/frozen_harness/analyze_replication.py` has a host name replaced in a default label (original sha256 `9136bd6bd39a30b83a363163cfdb2e90b1ceea69aa884decf0e4132bbf4b9e5b`, 24,529 bytes); the pins record the released file.

The archived notes and records in `inputs/` keep their original text, including the names of the AI assistants that reviewed them (see note 7 in `registrations/README.md`).
