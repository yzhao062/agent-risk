<div align="center">

# agent-risk: Risk Assessment for Changes to AI Agents

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-BSD_2--Clause-blue.svg)](https://github.com/yzhao062/agent-risk/blob/main/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![CI](https://github.com/yzhao062/agent-risk/actions/workflows/ci.yml/badge.svg)](https://github.com/yzhao062/agent-risk/actions/workflows/ci.yml)

</div>

Before changing a deployed AI agent (a stricter safety gate, a new model), organizations must approve the change, starting from existing logs. `agent-risk` extends classical risk analysis to partial records and partial changes to running systems: it replays logged episodes under the proposed change, assuming nothing else changes and the agent does not anticipate the change. The log settles episodes that act as before, a declared stop value settles stopped ones, and the rest need logged support, a saved-state rerun, or a bound. In a bash sandbox, a 7,200-episode log alone decided neither of two retry gates. Rerunning only each gate's blocked episodes from saved states reached both live decisions at half the agent calls of a live test sized post hoc for them, given the log. Without per-step harm, a stopping gate's certified harm ceiling rose up to 2.5-fold, since the log hides whether harm preceded the stop.

---

## Installation

```bash
# Install from GitHub:
pip install git+https://github.com/yzhao062/agent-risk
```

Or clone the repository and run `pip install .` at its root. The distribution name is `ai-agent-risk`; the import name is `agent_risk`.
Requires Python 3.9 or later, NumPy (>= 1.20), and SciPy (>= 1.7).

---

## Quickstart

The snippet below counterfactually replays three episodes under a stopping gate (theta = 0.94), computes replayed harm, failure, and cost, certifies an exact Clopper-Pearson upper bound on harm, and bounds harm under a retry gate:

```python
from agent_risk import Episode, certify_replay, replay_stop, retry_harm_bounds

scores, cost = [0.10, 0.20, 0.97, 0.30], [10, 20, 30, 40]
episodes = [
    Episode(scores, [False, True, False, False], cost, failed=False),  # harm at step 1
    Episode(scores, [False, False, False, True], cost, failed=False),  # harm at step 3
    Episode([0.10, 0.20], [False, False], [10, 20], failed=False),     # never above 0.94
]

replay = replay_stop(episodes, theta=0.94)
print(replay.branches, replay.stop_steps)  # ('stop', 'stop', 'agree') (2, 2, None)
print(round(replay.p_harm, 3))             # 0.333: only the step-1 harm survives
print(round(replay.p_failure, 3))          # 0.667: a stop counts as a failure by default
print(replay.mean_cost)                    # 50.0 = (60 + 60 + 30) / 3
print(round(replay.stop_mass, 3), round(replay.stop_clean_share, 3))  # 0.667 0.333

cert = certify_replay(replay, alpha=0.05)
print(cert.n_harmed, cert.n, round(cert.upper, 3))  # 1 3 0.865

bounds = retry_harm_bounds(episodes, theta=0.94, cap=2)
print([round(v, 3) for v in bounds.interval])       # [0.333, 0.667]
```

Logs can also be loaded directly from JSON Lines, with `null` indicating unobserved harm:

```python
from agent_risk import load_jsonl

lines = ['{"scores": [0.1, 0.97], "harm": [false, null], "cost": [5, 6], "failed": false}']
episodes = load_jsonl(lines)  # or load_jsonl("log.jsonl")
```

For paired policy comparisons and statistical equivalence testing:

```python
from agent_risk import is_equivalent, newcombe_paired_interval

ci = newcombe_paired_interval(both=40, first_only=1, second_only=2, neither=57, level=0.90)
print(round(ci.estimate, 3), round(ci.lower, 3), round(ci.upper, 3))  # -0.01 -0.043 0.023
print(is_equivalent(ci, margin=0.10))  # True: equivalent at alpha = 0.05
```

---

## Running Tests

To run the unit and regression test suite:

```bash
pip install ".[test]"
pytest
```

This executes all 149 tests in [`tests/`](https://github.com/yzhao062/agent-risk/tree/main/tests) (covering replay algorithms, stop valuations, bounds, and paper test vectors). If optional `statsmodels` is not installed, two comparison tests will be skipped (147 passed, 2 skipped).

---

## Repository Map

| Directory / File | Description |
|---|---|
| [`src/agent_risk/`](https://github.com/yzhao062/agent-risk/tree/main/src/agent_risk) | Core Python package: episode representation, stopping replay, retry bounds, exact Clopper-Pearson certification, and paired Newcombe confidence intervals. |
| [`tests/`](https://github.com/yzhao062/agent-risk/tree/main/tests) | Unit and regression test suite (149 tests) covering replay algorithms, stop valuations, bounds, and paper test vectors. |
| [`checks/`](https://github.com/yzhao062/agent-risk/tree/main/checks) | 15 finite-instance mathematical check scripts verifying theorems, combinatorial bounds, and design invariants across parameter spaces. |
| [`experiments/`](https://github.com/yzhao062/agent-risk/tree/main/experiments) | Experiment harness, execution runners, and analysis scripts for reproducing empirical evaluations. See [`RUNNING.md`](https://github.com/yzhao062/agent-risk/blob/main/experiments/RUNNING.md) for execution commands. |
| [`registrations/`](https://github.com/yzhao062/agent-risk/tree/main/registrations) | Registration documents for the paper's runs (sample sizes, analysis rules, margins, prediction hashes), each with its registration time; the paper and `ARTIFACT.md` mark which analyses were post hoc. |
| [`data/`](https://github.com/yzhao062/agent-risk/tree/main/data) | Log manifest, SHA-256 checksums, and download instructions for the experiment logs (hosted on Hugging Face). |
| [`survey/`](https://github.com/yzhao062/agent-risk/tree/main/survey) | The characteristic survey: 23 coded sources with their recorded versions, 30 AI Incident Database incidents, the 20-code codebook, both coders' matrices, and the agreement script. |
| [`supplement/`](https://github.com/yzhao062/agent-risk/tree/main/supplement) | Companion files for the paper's appendices (extended survey tables, characteristic and practice mappings, diagnostics), as submitted with the paper. |
| [`ARTIFACT.md`](https://github.com/yzhao062/agent-risk/blob/main/ARTIFACT.md) | Master artifact reproduction guide mapping paper names to run codes and exact reproduction commands. |

---

## Reproducing Paper Results

To reproduce the numbers, tables, and figures from the paper:
1. Consult [`ARTIFACT.md`](https://github.com/yzhao062/agent-risk/blob/main/ARTIFACT.md) for the master lookup table mapping paper run names to log codes, declared policies, monitor prompts, and exact command lines.
2. See [`experiments/README.md`](https://github.com/yzhao062/agent-risk/blob/main/experiments/README.md) for instructions on running offline log analysis and reproducing demonstration tables. For running live experiments, serving models, or re-executing harness runs (R0 to R13), see [`experiments/RUNNING.md`](https://github.com/yzhao062/agent-risk/blob/main/experiments/RUNNING.md).
3. See [`checks/README.md`](https://github.com/yzhao062/agent-risk/blob/main/checks/README.md) to run the 15 finite-instance verification scripts that validate the mathematical claims.

---

## Citation

```bibtex
@article{zhao2026agentrisk,
  title   = {agent-risk: Risk Assessment for Changes to AI Agents},
  author  = {Zhao, Yue and Xu, Zixiang and Zhang, Angela Huyue and Yang, S. Alex and Hu, Xiyang},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

---

## License

- **Code and Package**: The `agent-risk` software and library code in this repository are licensed under the [BSD 2-Clause License](https://github.com/yzhao062/agent-risk/blob/main/LICENSE).
- **Experiment Logs and Data**: The experimental logs, trajectory traces, analysis outputs, and manifests hosted on Hugging Face ([`yzhao062/agent-risk-logs`](https://huggingface.co/datasets/yzhao062/agent-risk-logs)) are released under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). See [`LICENSE-DATA.md`](https://github.com/yzhao062/agent-risk/blob/main/LICENSE-DATA.md).
- **Third-Party Components**:
  - **ControlArena**: Task definitions, environments, and bash tool harnesses (used in all bash-environment runs) derive from [ControlArena](https://github.com/UKGovernmentBEIS/control-arena), licensed under the [MIT License](https://opensource.org/licenses/MIT).
  - **$\tau^2$-bench**: Airline customer-support benchmark tasks, dialogue environments, and customer simulator code (used in the airline runs, E3 and E5) derive from [$\tau^2$-bench](https://github.com/sierra-research/tau2-bench), licensed under the [MIT License](https://opensource.org/licenses/MIT).
  - **Qwen Models**: The agent and monitor models (Qwen3-14B-FP8, Qwen3-14B, Qwen3-4B-Instruct-2507, Qwen3.5-27B-FP8, Qwen2.5-14B-Instruct) are released by their developers under the [Apache-2.0 License](https://www.apache.org/licenses/LICENSE-2.0). Neither this repository nor the dataset contains model weights.
