# Experiment Registrations and Protocol Records

This directory contains the protocol registration documents, pre-analysis plans, and execution records for the experimental runs reported in the paper:
**"agent-risk: Risk Assessment for Changes to AI Agents"** (arXiv).

As reported in the paper, eight of the nine registered runs fixed sample size, analysis rules, and equivalence margins before launch, and three froze predictions before new live records were evaluated. The documents in this directory comprise protocol registrations, versioned plans, and completed-run execution records. These documents record frozen manifest checksums (SHA-256), random seed schedules, declared loss definitions, analysis estimators (e.g., Newcombe hybrid score confidence intervals, Clopper-Pearson exact binomial bounds), equivalence margins, and prediction hashes.

Internal infrastructure identifiers (such as remote IP addresses, internal virtual machine names, HPC allocation codes, volume IDs, SSH configurations, and local user paths) have been replaced with neutral placeholders (e.g., `<HOST>`, `<IP>`, `<USER>`, `<ALLOCATION>`). All scientific rules, mathematical definitions, sample sizes, margins, thresholds, and cryptographic hashes are preserved verbatim.

---

## Mapping of Registration Documents to Paper Runs

| Registration File | Run Code(s) | Name in the Paper | Document Time & Type | Primary Estimands & Role |
|---|---|---|---|---|
| [`D58-PREREG-2026-10-04.md`](D58-PREREG-2026-10-04.md) | D58 (`B2`, `Val3`) | The registered replication (the replication log B2 and the live replication runs Val3) | 2026-10-04T07:24:37Z (UTC)<br>*(pre-registration)* | Equivalence test between live stopping gate at $\theta = 0.94$ and replayed pass-all log ($N = 1{,}000$ per arm, Newcombe 90% envelope inside $\pm 0.03$). |
| [`D64-PREREG-2026-10-04.md`](D64-PREREG-2026-10-04.md) | D64 | The paired announcement test | 2026-10-04T12:38:30Z (UTC)<br>*(revised 12:50 UTC; pre-registration)* | Within-task paired test of gate announcement in system prompt (400 triplets: baseline B, unannounced U, announced A; Newcombe method 10 paired interval). |
| [`D65-PREREG-2026-10-04.md`](D65-PREREG-2026-10-04.md) | D65 | The long-horizon log | 2026-10-04T17:13:26Z (UTC)<br>*(pre-registration)* | Long-horizon execution at step cap $K = 40$ ($N = 600$ baseline episodes); deployment vs. step-horizon readings and step-governed range. |
| [`D66-PREREG-2026-10-04.md`](D66-PREREG-2026-10-04.md) | D66 | The two-agent relay | 2026-10-04T18:40Z (UTC)<br>*(revised 19:25 UTC; pre-registration)* | Two-shift composition with handoff state; primary contrasts P1 (clean start without support), P2 (snapshot with support), P3 (independence foil); Fréchet bounds. |
| [`E1-PREREG-2026-10-05.md`](E1-PREREG-2026-10-05.md) | E1 | The targeted-execution run (retry gate) | 2026-10-05T01:10Z (UTC)<br>*(revised 01:40Z–04:00Z; pre-registration)* | Targeted continuation execution from container snapshots at the first blocked step to identify retry gate harm; Clopper-Pearson exact binomial bounds. |
| [`E2-PREREG-2026-10-05.md`](E2-PREREG-2026-10-05.md) | E2 | The stronger-agent run | 2026-10-05T00:10Z (UTC)<br>*(revised ~01:45Z; completed 18:57Z)* | Second operating point using Qwen3.5-27B-FP8; exploratory pilot for margin calibration followed by confirmatory batch ($N = 1{,}400$ pairs). |
| [`E3-PREREG-2026-10-05.md`](E3-PREREG-2026-10-05.md) | E3 | The airline run | 2026-10-05T02:40Z (UTC)<br>*(refrozen 05:30Z; completed 19:10Z)* | Framework verification in a second domain ($\tau^2$-bench airline customer-service benchmark, $N = 1{,}200$ pairs); customer simulator, state restoration without containers. |
| [`E5-PREREG-2026-10-06.md`](E5-PREREG-2026-10-06.md) | E5 | The airline retry gate by targeted execution | 2026-10-06T05:00Z–06:05Z (UTC)<br>*(pre-registration)* | Targeted continuation execution in airline environment via database state restoration; paired comparison with live retry gate ($N = 1{,}200$ units). |
| [`E7-PREREG-2026-10-06.md`](E7-PREREG-2026-10-06.md) | E7 | The two-gate run (two retry gates by targeted execution from one log) | 2026-10-06T16:20Z (UTC)<br>*(end time in Part B not stated; pre-registration)* | Large-scale change-agnostic log ($N = 7{,}200$ task draws; 36,000 planned episodes) evaluating two distinct retry gates ($\lambda_1 = 11.551292$ and $\lambda_2 = 10.25$) from shared baseline snapshots; stratified budgeted bounds with proportional allocation. |
| [`ARMS-CD-2026-10-04.md`](ARMS-CD-2026-10-04.md) | `B1`, `Arm C`, `Arm D` | The held-out log (`B1`), the live retry gate (`Arm C`), and the first announcement test (`Arm D`) | 2026-10-04 (times in UTC)<br>*(completed-run record)* | Held-out evaluation batch ($N = 300$ per arm); live retry gate comparison with log-alone bounds, initial prompt announcement intervention, and B1 baseline logging. |
| [`ARM-E-2026-10-04.md`](ARM-E-2026-10-04.md) | Arm E | The model-update reruns | 2026-10-03 to 2026-10-04 (DeltaAI clock, US Central)<br>*(completed-run record)* | Offline batch inference testing model updates across precision, kernel, and model changes (Rungs 1 to 4: Qwen3-14B FP8, BF16, Qwen3.5-27B, Qwen2.5-14B). |
| [`PLAN-EXPERIMENTS-2026-10-04.md`](PLAN-EXPERIMENTS-2026-10-04.md) | `B0`, `Arm A`, `N2 stage 1`, `D60` | The exploratory log (`B0`), exploratory live gates (`Arm A`), step-0 continuations (`N2 stage 1`), and monitor rescoring (`D60`) | Version 1: 2026-10-03 22:00 PDT<br>Version 2: 2026-10-04 05:45 UTC<br>Version 3: 2026-10-04 06:45 UTC<br>*(versioned plan)* | Foundational experiment plan governing exploratory pilot runs, decision rules, threshold calibration, and step-0 continuation feasibility. |

---

## Document Summaries and Scientific Roles

### 1. Replication and Announcement Tests
- **`D58-PREREG-2026-10-04.md`**: Registers the confirmatory replication batch on an independent draw of 2,000 episodes (1,000 baseline B2 and 1,000 live stopping gate Val3 at threshold $\theta = 0.94$), alternating in 20 balanced blocks. It sets the primary equivalence margin at 0.03 using the Newcombe 90% confidence envelope over all compatible missing-outcome completions.
- **`D64-PREREG-2026-10-04.md`**: Pre-registers the task-paired announcement test across 400 task triplets (1,200 episodes) comparing baseline (B), unannounced stopping gate (U), and announced stopping gate (A) where gate rules are revealed in the agent system prompt. Uses Newcombe paired difference method 10 with exact envelope optimization.

### 2. Horizon, Composition, and Agent Variation
- **`D65-PREREG-2026-10-04.md`**: Establishes protocol for extending the execution horizon to 40 steps on 600 baseline episodes. Defines the deployment reading (context/token limits treated as part of deployment) vs. step-horizon reading (censored at limit), and specifies the step-governed range.
- **`D66-PREREG-2026-10-04.md`**: Pre-registers the two-agent relay experiment ($N = 1{,}600$ tasks) testing component risk composition under partial state logging. Specifies primary contrasts P1 (clean start without support), P2 (snapshot start with support), and P3 (the independence foil $\beta - p_1 p_2$), along with sharp Fréchet bounds across declared state partitions.
- **`E2-PREREG-2026-10-05.md`**: Registers evaluation on a substantially more capable agent (Qwen3.5-27B-FP8), featuring a log-scale monitor scoring function $s = -\log_{10}(1 - \mathbb{E}[N]/100)$, exploratory pilot for margin calibration, and confirmatory run on 1,400 paired units.

### 3. Targeted Execution and Budgeted Certification
- **`E1-PREREG-2026-10-05.md`**: Documents targeted continuation execution where only episodes blocked by the retry gate are resumed from container snapshots. Pre-registers the prediction SHA-256 hash before running live comparisons, demonstrating exact identification of retry risk with a fraction of live computation.
- **`E3-PREREG-2026-10-05.md`**: Pre-registers extension to the airline customer-support environment ($\tau^2$-bench airline v1.0.1, 1,200 paired episodes), verifying framework generalizability across distinct tool interfaces, state representations, and customer dialogue.
- **`E5-PREREG-2026-10-06.md`**: Registers targeted execution in the airline domain via database replay and state restoration, testing whether container-free state reconstruction reproduces live retry risk.
- **`E7-PREREG-2026-10-06.md`**: Pre-registers the large-scale two-gate demonstration ($N = 7{,}200$ units, 36,000 planned episodes) evaluating two retry gates ($\lambda_1 = 11.551292$ and $\lambda_2 = 10.25$) simultaneously from change-agnostic snapshots. Pre-registers prediction SHA-256 `a3bccf79...` and establishes stratified budgeted bounds with proportional allocation for certification.

### 4. Foundational & Multi-Arm Baselines
- **`ARMS-CD-2026-10-04.md`**: Records the execution and outcomes of Arms C (live retry gate) and D (announcement test) alongside held-out baseline log B1 ($N = 300$ each).
- **`ARM-E-2026-10-04.md`**: Records offline batch inference reruns testing model changes (FP8, BF16, model family rollback to Qwen2.5, and model update to Qwen3.5) on logged decisions.
- **`PLAN-EXPERIMENTS-2026-10-04.md`**: The master experimental plan setting out the mathematical formulation, loss functions, gate definitions, sample size rationale, and decision registers for exploratory runs (B0, Arm A, N2, D60).

---

## Sanitization Notes

All files in this directory have been sanitized for public release:
1. **IP Addresses & Subnets**: Remote public IP addresses have been replaced with placeholders such as `<IP>`, `<IP_1>`, `<IP_2>`, `<IP_A>`, `<IP_B>`, and Docker container subnets have been replaced with `<DOCKER_SUBNET>`. Local loopback addresses (`127.0.0.1`) are retained where they describe local proxy ports.
2. **Hostnames, VMs & Nodes**: Internal cloud instance names, VM identifiers, and Slurm node identifiers have been replaced with `<HOST>`, `<HOST_1>`, `<HOST_2>`, `<HOST_A>`, `<HOST_B>`, `<PREVIOUS_HOST>`, `<NODE_1>`, etc. Placeholder numbering (such as `<HOST_1>`, `<IP_1>`) is assigned independently per file.
3. **Allocations & Project Codes**: HPC allocation codes and Slurm account names have been replaced with `<ALLOCATION>`, `<ALLOCATION_OTHER>`, `<ACCOUNT>`, and `<ACCOUNT_SECONDARY>`.
4. **Cloud Volume, Image & Instance Identifiers**: UUIDs and labels identifying cloud instances, disk volumes, SSH configurations, and snapshots have been replaced with `<INSTANCE_ID>`, `<INSTANCE_ID_1>`, `<VOLUME_ID_1>`, `<VOLUME_LABEL>`, `<IMAGE_ID>`, `<SHELVED_IMAGE>`, and `<SSH_CONFIG>`.
5. **Local Paths, Usernames & Personal Names**: Absolute paths referencing local user directories and usernames have been replaced with repository-relative paths, `$HOME`, `<USER>`, or `<HOME_PATH>`, and personal first names have been replaced with neutral roles ('author' or 'coordinator'). Absolute interpreter paths and `H=` environment prefixes have been removed.
6. **Registration Scope & Integrity of Scientific Content**: The registration files in this directory represent the complete, final protocol documents (including launch verification notes and post-launch records where applicable); frozen pre-launch copies of registrations launched in the field are preserved with their original commitments under `logs/<run>/` in the dataset. Certain registration documents reference internal run codes for runs excluded from this paper (such as E6 in the E7-PREREG seed scan); all seed ranges, registered sample sizes ($N$), random seeds, threshold parameters ($\theta$, $\lambda$), confidence levels, statistical estimators, margins ($\delta$), and SHA-256 prediction/manifest hashes remain completely unchanged.
7. **Review Notes**: The registrations and records keep their original text about how each was checked before launch. They name the AI assistants that reviewed them as independent checkers (Codex, Antigravity, and Claude) and cite review files and working notes in an internal archive (paths under `docs/agent-risk-modeling/` or `agent-io/`) that is not released. The frozen launch copies in the dataset carry the same text, so it stays as written.
