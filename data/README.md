# Experiment Logs and Data Manifests

This directory contains the registration manifests, file manifests, and download instructions for the experimental logs supporting the paper:

> **agent-risk: Risk Assessment for Changes to AI Agents**  
> Research paper (arXiv preprint)

The experimental logs total **3,305.33 MiB** (3,465.89 MB uncompressed payload across 2,495 analysis-sufficient files). Because of their size, the logs are hosted separately from the Git repository.

---

## 1. Hosting

The logs are hosted on Hugging Face Datasets:

- **Repository**: [`yzhao062/agent-risk-logs`](https://huggingface.co/datasets/yzhao062/agent-risk-logs)
- **Dataset Card**: See the dataset card on Hugging Face or in the release repository for full per-run breakdowns, Inspect archive reading snippets, and citation information.

---

## 2. Directory Layout and Repository Organization

The repository separates small, reproducible manifests (tracked in Git) from bulk experiment logs (hosted externally on Hugging Face):

```
agent-risk/
├── data/
│   ├── README.md               # This document (layout, hosting, and verification recipe)
│   ├── MANIFEST.tsv            # TSV manifest: path, byte size, released SHA-256, original SHA-256 (2,495 files)
│   ├── REDACTIONS.tsv          # TSV record: path, original and released SHA-256, rules, counts, notes (58 files)
│   └── manifests/              # Per-run pre-registration manifests and code checksums
│       ├── d58/                # Registered replication (B2 and Val3) manifest
│       ├── d64/                # Paired announcement test manifest
│       ├── d65/                # Long-horizon manifests (combined, VM2, VM3)
│       ├── d66/                # Two-agent relay manifest, task classes, and SHA256SUMS
│       ├── e1/                 # Targeted-execution run manifest and SHA256SUMS
│       ├── e2/                 # Stronger-agent run manifest and SHA256SUMS
│       ├── e3/                 # Airline customer-service run manifest and SHA256SUMS
│       ├── e5/                 # Airline retry gate targeted-execution manifest, units, and extract
│       └── e7/                 # Two-gate run manifest and SHA256SUMS
└── experiments/                # Experiment harness code
    └── analysis/               # Analysis scripts reproducing paper numbers
```

### Destination Log Layout (`data/`)

When downloaded to `local_dir="data"`, the archive populates the following directory tree expected by the analysis commands in `ARTIFACT.md`:

```
data/
├── logs/
│   ├── pilot-ab/
│   ├── arms-cd/
│   ├── d58/
│   ├── d64/
│   ├── d65/
│   ├── d66/
│   ├── e1/
│   ├── e2/
│   ├── e3/
│   ├── e5/
│   ├── e7/
│   └── n2/
└── arm-e/
    ├── data/
    └── results/
```

Commands documented in `ARTIFACT.md` and `experiments/` use explicit repository-relative paths (`data/logs/...`, `data/arm-e/...`). Scripts that use `AGENT_RISK_LOGS` default to `data/logs`.

| Paper Name | Run Code | Directory Path | Key Contents | Read by Analysis Commands |
|---|---|---|---|---|
| The exploratory log | B0 | `logs/pilot-ab/b0/` | `episodes.jsonl`, `plan.json`, `run.json` | A8 |
| The exploratory live gates | Arm A | `logs/pilot-ab/val2_t1/` to `val2_t3/` | `episodes.jsonl`, `plan.json`, `run.json` | A8 |
| The held-out log | B1 | `logs/arms-cd/b1/` | `episodes.jsonl`, `*.eval`, `plan.json`, `run.json` | A1, A4, A5, A7, A8, Tab D.3 |
| The live retry gate | Arm C | `logs/arms-cd/c_retry/` | `episodes.jsonl`, `plan.json`, `run.json` | A7, A8 |
| The first announcement test | Arm D | `logs/arms-cd/d_announce/` | `episodes.jsonl`, `plan.json`, `run.json` | A8 |
| Held-out and pooled bounds | - | `logs/arms-cd/` | `retry_bounds_b0.json`, `retry_bounds_b1.json`, `retry_bounds_pooled.json`, `analysis_arms_cd.json` | A7, A8 |
| The registered replication (the replication log and the live replication runs) | D58 (B2 / Val3) | `logs/d58/` | 20 blocks (`block-01-B2` to `block-20-Val3`) with `episodes.jsonl`, `analysis_d58.json` | A1, A5, A28 |
| The paired announcement test | D64 | `logs/d64/` | 12 blocks (`block-01-B` to `block-12-A`) with `episodes.jsonl`, `analysis_d64.json` | A2 |
| The step-0 continuations | N2 stage 1 | `logs/n2/stage1/` and `logs/n2/` | `episodes.jsonl`, `serving_check.json`, `context_check.json`, `logs/n2/analysis_n2.json` | A7 |
| Held-out monitor scores recomputed (Table D.3; 'monitor rescoring' in logs) | D60 | `logs/n2/d60/` | `d60_rescore.json` | Tab D.3 |
| The model-update reruns | Arm E | `arm-e/` | `data/` and `results/` across 4 update rungs (r1_same, r2_bf16, r3_qwen35, r4_qwen25) | A6, Tab D.3 |
| The long-horizon log | D65 | `logs/d65/` | `vm2/` and `vm3/` blocks, `analysis_d65.json`, `incident_vm2.json` | A9, A10, A11 |
| The two-agent relay | D66 | `logs/d66/` | 102 blocks (`a01-a51`, `b01-b51`), `watchdog_*.jsonl`, 102 `.eval` files, `analysis_d66.json` | A12, A13, A25 |
| The targeted-execution run (retry gate) | E1 | `logs/e1/` | 42 blocks (`a01-a21`, `b01-b21`), `units.json`, `watchdog_*.jsonl`, 42 `.eval` files, `prediction_e1.json`, `analysis_e1.json` | A14, A15, A16, A23, A24, A27, A28 |
| The stronger-agent run | E2 | `logs/e2/` | 28 blocks with `episodes.jsonl`, `watchdog_h1.jsonl`, `analysis_e2.json` | A17, A26, A28 |
| The airline run | E3 | `logs/e3/` | 24 blocks with `episodes.jsonl`, `recompute.json`, `analysis_e3.json`, `report_e3.json` | A18, A19, A20, A22 |
| The airline retry gate by targeted execution | E5 | `logs/e5/` | 8 blocks (`block-01-h1` to `block-08-h1`) with `episodes_live.jsonl` and `episodes_cont.jsonl`, `prediction_e5.json`, `analysis_e5.json` | A21, A22 |
| The two-gate run (two retry gates by targeted execution from one log) | E7 | `logs/e7/` | 150 blocks (`h1-001` to `h6-025`) with `episodes_{base,cont,live}.jsonl`, `prediction_e7.json`, `analysis_e7.json`, `posthoc_*.json` | A29 through A39 |

---

## 3. Download and Verification Recipe

### Step 1: Download from Hugging Face

Using Python `huggingface_hub`:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="yzhao062/agent-risk-logs",
    repo_type="dataset",
    local_dir="data",
    allow_patterns=["logs/*", "arm-e/*", "MANIFEST.tsv", "REDACTIONS.tsv"],
)
```

The patterns skip the dataset card (the dataset's own README.md), so a download into a clone of the GitHub repository leaves the repository's own `data/README.md` unchanged; `MANIFEST.tsv` and `REDACTIONS.tsv` are byte-identical to the copies in the repository.

Or using the Hugging Face CLI:

```bash
# Using modern hf CLI:
hf download yzhao062/agent-risk-logs --repo-type dataset --local-dir data --include "logs/*" "arm-e/*" MANIFEST.tsv REDACTIONS.tsv

# Or using huggingface-cli:
huggingface-cli download yzhao062/agent-risk-logs --repo-type dataset --local-dir data --include "logs/*" "arm-e/*" MANIFEST.tsv REDACTIONS.tsv
```

### Step 2: Configure Environment Variable

`AGENT_RISK_LOGS` defaults to `data/logs` inside the repository:

```bash
export AGENT_RISK_LOGS=$(pwd)/data/logs
```

### Step 3: Verify Integrity with SHA-256

Verify file count, byte sizes, and checksums against `MANIFEST.tsv` using Python:

```python
import hashlib
from pathlib import Path

data_root = Path("data")
manifest_path = Path("data/MANIFEST.tsv")

mismatches, missing, total = [], [], 0
with open(manifest_path, "r", encoding="utf-8") as f:
    next(f)  # skip header
    for line in f:
        total += 1
        path_str, byte_str, expected_sha, *_ = line.strip().split("\t")
        target = data_root / path_str
        if not target.exists():
            missing.append(path_str)
            continue
        data = target.read_bytes()
        if len(data) != int(byte_str) or hashlib.sha256(data).hexdigest() != expected_sha:
            mismatches.append(path_str)

print(f"Verified {total} files: {total - len(missing) - len(mismatches)} OK, {len(missing)} missing, {len(mismatches)} mismatched.")
```

---

## 4. Redaction Policy in Brief

All logs in the release dataset have been sanitized to remove private and infrastructure identifiers while keeping all experimental and scientific data intact:
- Cloud VM public IPv4 addresses are replaced with `<vm-ip>`, and internal VM hostnames with `vm-<n>`.
- Delta HPC cluster node names are replaced with `<hpc-node>`.
- Local user home directory paths are replaced with `<home>`, and usernames with `<user>`.
- Paths under `/home/ubuntu/`, the default user of the cloud VM image, are kept; they name no person or machine.
- Cloud project and allocation codes are replaced with `<allocation>`.
- SSH connection strings are replaced with `ubuntu@<vm-ip>`.
- Cloud instance UUIDs are replaced with `<instance-id>`.

### Hash-Preserved Pre-Registered Files
Six pre-computed analysis and prediction records contain only a local home directory string and remain byte-identical to their pre-registered versions to allow external hash verification against registration records:
- `logs/arms-cd/retry_bounds_b0.json`
- `logs/arms-cd/retry_bounds_b1.json`
- `logs/arms-cd/retry_bounds_pooled.json`
- `logs/e1/analysis_e1.json`
- `logs/e5/prediction_e5.json`
- `logs/e7/analysis_e7.json`

In addition, the two registered E5 manifests under `data/manifests/e5/` (`e5_units.json` and `e5_b_extract.json`) contain only a local home path and retain their registered bytes so their pre-registered SHA-256 digests (`30096b7a8d76...` and `dbc5395eae17...`) remain verifiable.

### Redacted Pre-Registration Launch Copies
The launch copies of registrations for runs E3, E5, and E7 (`logs/e3/E3-PREREG-2026-10-05.launched.md`, `logs/e5/E5-PREREG-2026-10-06.launched.md`, and `logs/e7/E7-PREREG-2026-10-06.launched.md`) are redacted in the release to remove VM IPs, host names, and instance codes. Their original SHA-256 digests remain recorded in their respective launch records, pre-registrations, and `REDACTIONS.tsv`; unredacted originals are available to reviewers on request.

### Pre-Registration Checksums and Unreleased Records
The `SHA256SUMS` files under `data/manifests/` are the registered original checksum lists for full run directories recorded at pre-registration and launch time. They list files of the full run directories, including intermediate developmental code snapshots and local environment captures, some of which are not released in the public dataset. Furthermore, code checksums in `SHA256SUMS` and in analysis records refer to pre-release scripts prior to release sanitization. Similarly, unreleased run directories and the per-host serving records (`serving_host_*.json`) of most runs are not included; the release keeps the two serving records of the E1 run (read by A23), the one of the held-out log (read by R6), and the one of the step-0 continuations, with host names redacted (see `REDACTIONS.tsv`).
