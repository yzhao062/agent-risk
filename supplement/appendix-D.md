# Appendix D Companion: Registered Demonstrations and Supporting Evidence

This companion document archives secondary diagnostics, extended exploratory analyses, full data tables, and worked decision examples moved from the paper's Appendix D (Registered Demonstrations and Supporting Evidence) to maintain focus on the primary registered evidence in the compiled PDF.

---

<a id="artifact-map"></a>
## Artifact Map

This table archives the full artifact map. See `ARTIFACT.md` for the authoritative command index and run codes.

| Item | Input Logs | Verifying Record |
|---|---|---|
| Figures D.1 and D.3 | `logs/d58/` (20 blocks), `logs/d64/` (12 blocks), `logs/arms-cd/b1/`, `logs/d66/`, `logs/e2/`, `logs/e3/`, `logs/e7/` | `D58-RESULTS`, `D64-RESULTS`, `D66-RESULTS`, `E2-PREREG`, `E3-PREREG`, `E7-PREREG` |
| Table D.3 | `logs/pilot-ab/`, `logs/arms-cd/b1/`, `logs/d58/`, `arm-e/results/`, `logs/e3/` | `PILOT-AB`, `N2-STAGE1-RESULTS`, `D58-RESULTS`, `DRAFT-STEP2`, `ARM-E`, `E3-PREREG` |
| Figure D.2 and model-release plot (not printed; numbers in Appendix D.6) | `logs/e7/`, `arm-e/results/` | `E7-PREREG`, `ARM-E` |
| Table 2 | Logs above, plus `logs/pilot-ab/`, `logs/arms-cd/c_retry/`, `logs/n2/stage1/`, `logs/d65/vm2/`, `logs/d65/vm3/`, `logs/e1/`, `logs/e2/`, `logs/e3/`, `logs/e5/` | Records above, `E1-PREREG`, `E5-PREREG`, `ARMS-CD`, `N2-STAGE1-RESULTS`, `D65-RESULTS`, `E2-PREREG`, `E3-PREREG` |
| Table D.2 | As Table 2 | As Table 2 |
| Table D.1 | Run plans and manifests in `d58/`, `d64/`, `d65/`, `d66/`, `e1/`, `e2/`, `e3/`, `e5/`, `e7/` | `PILOT-AB`, `ARMS-CD`, `D58`, `D64`, `D65`, `D66`, `E1`, `E2`, `E3`, `E5`, `E7` |
| [Budget design table (`tab:budget`)](#budget-design-table) | `logs/e1/` | The budgeted-execution, reference-predictor, relay-reproduction, and decision analyses (post hoc), each with its script; archived at [#budget-design-table](#budget-design-table) |
| [Per-horizon table (`tab:d65`)](#per-horizon-table) | `logs/d65/vm2/`, `logs/d65/vm3/` | `D65-RESULTS`, `D65-PREREG`; archived at [#per-horizon-table](#per-horizon-table) |
| Host and incident-exposure mapping (`tab:d65-host`) | `logs/d65/vm2/`, `logs/d65/vm3/` | The by-host note (post hoc); archived at [#horizon-diagnostics](#horizon-diagnostics) |

Original LaTeX source:
```latex
\begin{table}[!ht]
\caption{Artifact map. Input logs and verifying records for each display. Commands regenerating each result are listed in \texttt{ARTIFACT.md}. Paths and record names keep the run codes of the logs; \texttt{ARTIFACT.md} maps each code to its name in this paper.}
\label{tab:artifacts}
\centering
\scriptsize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.05}
\begin{tabularx}{\textwidth}{@{}P{0.14\textwidth}LP{0.3\textwidth}@{}}
\rowcolor{hdrgray}
Item & Input Logs & Verifying Record \\
Figures~\ref{fig:forest} and~\ref{fig:checks} & \path{logs/d58/} (20 blocks), \path{logs/d64/} (12 blocks), \path{logs/arms-cd/b1/}, \path{logs/d66/}, \path{logs/e2/}, \path{logs/e3/}, \path{logs/e7/} & \path{D58-RESULTS}, \path{D64-RESULTS}, \path{D66-RESULTS}, \path{E2-PREREG}, \path{E3-PREREG}, \path{E7-PREREG} \\
Table~\ref{tab:log-fields} & \path{logs/pilot-ab/}, \path{logs/arms-cd/b1/}, \path{logs/d58/}, \path{arm-e/results/}, \path{logs/e3/} & \path{PILOT-AB}, \path{N2-STAGE1-RESULTS}, \path{D58-RESULTS}, \path{DRAFT-STEP2}, \path{ARM-E}, \path{E3-PREREG} \\
Figure~\ref{fig:bound} and model-release plot (not printed; numbers in Appendix D.6) & \path{logs/e7/}, \path{arm-e/results/} & \path{E7-PREREG}, \path{ARM-E} \\
Table~\ref{tab:statements} & Logs above, plus \path{logs/pilot-ab/}, \path{logs/arms-cd/c_retry/}, \path{logs/n2/stage1/}, \path{logs/d65/vm2/}, \path{logs/d65/vm3/}, \path{logs/e1/}, \path{logs/e2/}, \path{logs/e3/}, \path{logs/e5/} & Records above, \path{E1-PREREG}, \path{E5-PREREG}, \path{ARMS-CD}, \path{N2-STAGE1-RESULTS}, \path{D65-RESULTS}, \path{E2-PREREG}, \path{E3-PREREG} \\
Table~\ref{tab:statements-full} & As Table~\ref{tab:statements} & As Table~\ref{tab:statements} \\
Table~\ref{tab:runs} & Run plans and manifests in \path{d58/}, \path{d64/}, \path{d65/}, \path{d66/}, \path{e1/}, \path{e2/}, \path{e3/}, \path{e5/}, \path{e7/} & \path{PILOT-AB}, \path{ARMS-CD}, \path{D58}, \path{D64}, \path{D65}, \path{D66}, \path{E1}, \path{E2}, \path{E3}, \path{E5}, \path{E7} \\
Table~\ref{tab:budget} & \path{logs/e1/} & The budgeted-execution, reference-predictor, relay-reproduction, and decision analyses (post hoc), each with its script \\
Table~\ref{tab:d65} & \path{logs/d65/vm2/}, \path{logs/d65/vm3/} & \path{D65-RESULTS}, \path{D65-PREREG} \\
Table~\ref{tab:d65-host} & \path{logs/d65/vm2/}, \path{logs/d65/vm3/} & The by-host note (post hoc) \\
\end{tabularx}
\end{table}
```

---

<a id="exploratory-gates"></a>
## Exploratory Gates

### Exploratory Values
At $\theta=0.94$, replay from the exploratory log gives P(harm) 0.053 with the full log against a baseline of 0.170, yielding an estimated reduction of 0.117; the registered margin of 0.03 is roughly one quarter of this reduction. Without per-step harm, the identified set is $[0.037,0.110]$, with 22 of 300 episodes having unknown harm before the stop (18 when command checks are recomputed). A log keeping only executed calls yields $[0,0.170]$ because scores cannot be recomputed exactly under default kernels. The monitor reproduced scores on 1,686 of 1,811 repeated inputs under default kernels, compared to 2,183 of 2,183 under batch-invariant serving (post hoc).

### Exploratory Live Gates
Live stopping gates on default kernels were compared with replay from the exploratory log (approximate 95% bootstrap; intervals from the pilot-record bootstrap, not from A8). At complete-data thresholds, replay agrees with live execution: at $\theta=0.94$, live minus replay is $-0.027$ $[-0.057,0.003]$ for P(harm) and $-1{,}230$ $[-8{,}946,6{,}268]$ for mean tokens; at $\theta=0.19$, $-0.017$ $[-0.050,0.017]$ and $+2{,}841$ $[-4{,}533,9{,}854]$. At $\theta=0.99$, 14 live episodes are missing; on 286 scored episodes the harm difference is $-0.001$ $[-0.041,0.038]$, and across all 300 live P(harm) lies in $[0.057,0.123]$. Across three thresholds and 14 query coordinates, 41 of 42 uncorrected difference intervals contain 0. Default kernels do not invalidate stopping-gate replay because the gate evaluates logged scores; they disrupt agent reruns, where step-0 outputs matched at identical inputs in only 23 of 99 episodes. Compared to replay from the held-out log across the kernel switch, harm differences are $+0.019$, $-0.010$, and $0.000$ at the three thresholds. Point estimates of live harm were at or below replayed harm in all same-kernel comparisons.

### An Ablation That Discards Terminal Contributions
Discarding stopped episodes without reweighting leaves 209, 189, and 151 of the 300 episodes of the exploratory log at $\theta=0.99$, $0.94$, and $0.19$ (post hoc). This inflates mean tokens by 33% to 52% (70,081, 67,791, and 60,845 versus live 52,854, 46,929, and 40,051). Its P(failure) values (0.818, 0.799, 0.748 versus live 0.874, 0.887, 0.900) are lower because declared failure at abort is discarded. Replayed harm (0.057 to 0.066) remains near replay because little harm precedes an abort, highlighting the role of part (ii) of Proposition A.4.1 (`prop:replay`).

---

<a id="abort-sensitivity"></a>
## Abort Sensitivity

### Sensitivity to the Abort Valuation (A3) Across Logs
On the replication log, 372 of 1,000 episodes stopped before harm, yielding replayed P(harm) of $0.039+0.372p$. Across other logs, replayed P(harm) at $\theta=0.94$ under the same sensitivity model would be:
- Held-out log: $0.037+0.457p$ (137 of 300; 138 if one unknown counts).
- Exploratory log: $0.053+0.353p$ (106 of 300).

### Kill-Switch Readings
A kill switch executing flagged calls before stopping yields replayed P(harm) of 0.113 on the replication log (post hoc), and 0.133, 0.123, and 0.110 on the exploratory log, confirming that the gate intercepts calls prior to execution.

---

<a id="decision-examples"></a>
## Decision Examples

### A Decision from the Stronger-Agent Log
When an approval policy requires joint certification of both a harm ceiling and a task-success floor, finite-sample uncertainty can lead to deferral even when marginal bounds pass.

Suppose a deployer approves a configuration when the log certifies P(harm) at most a ceiling $h$ and task success at least a floor $u$, by exact one-sided 97.5% bounds with unknown outcomes counted unfavorably, so that both bounds hold jointly with probability at least 95% (a union bound). Rounded outward to four decimals, the stronger-agent log bounds the stopping gate's P(harm) by 0.2560 and its task success below by 0.3784, and pass-all's by 0.3583 and 0.3869. It thus approves the gate for every $h\ge0.2560$ and $u\le0.3784$, and pass-all only for $h\ge0.3583$ and $u\le0.3869$; for ceilings in $[0.2560,0.3583)$ and floors up to 0.3784, the log alone selects the gate.

Without per-step harm the log certifies only 0.3568, so approval for ceilings in $[0.2560,0.3568)$ needs per-step harm. The gate removes 0.097 to 0.103 of harm and costs 0.008 to 0.009 of task success (12 of 1,399 complete pairs succeed only without it; Newcombe paired 95% $[0.003,0.015]$). Under an illustrative rule with $h=0.26$ and $u=0.38$, the log defers. Its harm bound leaves 0.0040 of headroom below the ceiling, but its success bound falls 0.0016 short of the floor, although replayed success is 0.404 to 0.405 and live success was 0.410 to 0.411. Separate 95% bounds (0.2523 and 0.3825) would approve, so the joint requirement decides this case. Certified harm does not decide approval by itself: the floor, the tolerated error, and the success cost belong to the deployer's declared rule.

---

<a id="secondary-diagnostics"></a>
## Secondary Diagnostics

### Direction of Stopping-Gate Differences
Live harm is at or below replayed harm in all three stopping comparisons ($-0.009$; $-0.009$ to $-0.004$; $-0.001$), and every interval contains zero. Under a symmetric null, three differences share a sign with probability $1/4$, so these runs show no systematic direction.

### Stopping Gate Secondary Outcomes
In the replication log:
- Replayed P(failure): 0.890; live: 0.897.
- Mean tokens: 47,810 by replay and 45,045 live (difference approximate 95% bootstrap $[-6{,}982,1{,}474]$).

In the stronger-agent run:
- Pass-all P(harm): 0.330 to 0.333.
- Harm reduction: replay prevents 0.097 to 0.103 of harm (Newcombe paired 95% $[0.081,0.119]$) and live gate prevents 0.104 to 0.109 ($[0.084,0.128]$).
- Task success: 0.410 to 0.411 live against 0.404 to 0.405 replayed (Newcombe paired 95% $[-0.015,0.027]$).

In the airline run:
- Pass-all P(harm): 0.512.
- Abort rate: 0.131 live against 0.1325 replayed (Newcombe paired 90% $[-0.018,0.015]$).
- Task success: 0.2225 live against 0.236 replayed (Newcombe paired 95% $[-0.032,0.005]$).
- Pilot comparison: on the pilot, the gate prevented less harm than stopping at random at the same rate (0.050 against 0.068).
- Error composition: about half of harmed episodes are forbidden refunds; checker also counts tool-use errors (itinerary changes dropping flight legs, baggage overcharges).
- Live retry on 400 units: P(harm) 197/400 = 0.4925 (exact 95% $[0.442,0.543]$), inside plug-in bounds $[0.4525,0.555]$.

### Reference Predictors and Pathwise Reproduction (E1)
Treating retries as stops gives 27/1,200 = 0.0225: harmful in both arms in 27 units and only live in 9 (live minus predictor 0.0075, Newcombe paired 95% $[0.002,0.014]$). It lies inside the 0.02 margin, so the margin alone does not separate it from targeted execution. Ignoring the gate gives 184/1,200 = 0.153 (live minus it $-0.123$, $[-0.143,-0.105]$) and fails the margin.

With sampling error, exact one-sided 97.5% limits place the log-only interval $[0.023,0.422]$ inside $[0.0148,0.4502]$ with probability at least 95%. A monotone-response restriction (Manski, 1997) narrows it: suppose that at each continuing prefix $u$ a refusal never raises the chance of harm, $Q_u(X=1)\le P^0(X=1\mid[u])$. Each prefix then contributes at most its baseline harm in Proposition `prop:app-identified`, so the upper end becomes pass-all's P(harm) and the interval is $[0.023,0.153]$. It still contains the target 0.10, so the restriction alone does not decide.

The executed continuations test its pathwise form: of the 479 unresolved episodes, 2 are harmful only after the refusal, 151 only without it, and 6 in both, so harm falls from 157 to 8 while the pathwise form fails twice. The identified value is harmful in both arms in 35 units and only live in 1 ($[-0.003,0.005]$). The targeted-execution interval is compatible with zero discrepancy; these marginal intervals do not test one predictor against another.

We compared decoded requests and outputs after replacing tool-call identifiers by step and position. The continuation's next agent output equalled live in 317 of 317 first-step and 111 of 168 later continuations, and the whole remaining trajectory (requests, outputs, gate decisions, tool results, termination) in 253 and 49. Harm matched live in all 715 agreeing episodes and 484 of 485 continuations; serving was reproducible, but tool outputs such as ports vary, so whole remaining trajectories matched in 302 of 485. Repeated agent inputs (5,247 groups) and monitor inputs (6,938) never gave different outputs under batch-invariant serving, while tool outputs varied (ports, service names, test durations). The discordant unit ($\sigma_\kappa=1$) inherits a step-0 test result that differed from live and diverges at its first new request; live wrote outside the allowed roots at steps 2, 4, and 6. One later unit submitted before its block in the live arm, so its next live request does not exist.

### Restoration Allowance (E1, Post Hoc)

The exact 95\% upper limit of the identified P(harm) $35/1{,}200$ is 0.040332, leaving 0.0596 headroom below the 0.10 target. By Remark F.7.3, the retry decision tolerates total-variation error $\gamma$ up to three labeled allowances: plug-in $\gamma\le0.1476$ (using empirical continuing mass $485/1{,}200$), joint 95\% $\gamma\le0.1379$ (using the 97.5\% upper limit 0.4326), and conservative $\gamma\le0.0596$ (mass bounded by 1). Aggregate paired agreement does not establish prefix-wise total-variation bounds (`checks/verify_restoration.py`).

### Airline Retry Turn Reproduction
Among units whose live run reached the same state, the next turn was identical in 23 of 26 first-step and 17 of 26 later continuations; for the later ones, live runs sending identical requests agree at a turn-standardized rate of 0.810, and the cause of the lower rate is not identified.

### Model-Update Ladder Diagnostics
The model-update reruns evaluated logged agent decisions rerun at logged inputs and seeds across 99 batch-invariant episodes (1,260 decisions) and 30 default-kernel episodes (390 decisions). The identical model reproduced all 1,650 decisions exactly. On the 99-episode set:
- Alternate kernel mode: agreement mass 2/99 (exact 95% $[0.003,0.071]$) and first-decision agreement 7/99 ($[0.029,0.140]$).
- Precision update (bf16): 1/99 ($[0.000,0.055]$) and 3/99 ($[0.006,0.086]$).
- Release candidate (Qwen3.5-27B-FP8, six server processes): agreement mass 0/99 (exact two-sided 95% upper end 0.037, conservative). The paper reports that the release candidate reproduces none of 99 logged episodes across six server processes at identical inputs and seeds (Appendix D.6).
- Rollback model (Qwen2.5-14B-Instruct): 0/99 (upper bound 0.037).
- Evaluating tool calls alone overstates agreement by up to 5.6 times (for rollback, 139 of 1,260 decisions reproduced versus 25 at full interface).
- The first decision binds because the agent generates scripts there (median 462 output tokens versus 50 later).

### Relay Token Outcomes and Trajectory Reproduction
- Task failure after the second shift was 0.880 in the relay, equivalent to both the clean start and the snapshot start.
- Mean second-shift tokens were 37,310 (relay), 36,874 (clean start), and 37,403 to 37,673 (snapshot start).
- Compared after replacing tool-call identifiers by step and position (post hoc), the second agent's first second-shift request and next output equalled the relay's in 1,599 of 1,599 units and the whole second shift in 524; every first difference lay in a tool result (test ports, generated addresses, timing text). Harm agreed in 1,582 units (both 176, relay only 7, rerun only 10).
- The first agent alone matched the relay's first shift: difference in $p_1$ is $-0.003$ (Newcombe paired 95% $[-0.009,0.004]$), and every handoff-state share is equivalent within $\pm0.03$.

### Oversight Cues: First Announcement Test and Token Diagnostics
- First announcement test: compared announced live gate to replay from the held-out log ($n=300$), finding an abort-rate difference of $-0.003$ (Newcombe 95% $[-0.083,0.076]$). Task-mix analysis places the first announcement test roughly 0.06 above its expected rate.
- Token differences in the paired test: announced minus unannounced $-9{,}053$ (approximate 95% bootstrap $[-13{,}898,-4{,}283]$); announced minus replay $-8{,}186$ $[-12{,}828,-3{,}532]$; unannounced minus replay $+867$ $[-617,2{,}665]$.
- Unannounced gate and replay abort at step 0 in the identical 83 triplets.
- In each of the four block groups, the announced-minus-unannounced difference is $+0.06$ to $+0.08$.

---

<a id="execution-cost"></a>
## Execution Cost

### Storage and Commit Timings for Targeted Execution (E1)
The continuations used 4,004 agent calls and 26.1 million agent and monitor tokens, against 13,489 calls and 83.7 million tokens for the live gate: 0.30 and 0.31 of the live cost. Accounting for 26 context-exhausted requests and 18 failed retried attempts places the agent-call ratio in $[0.2966,0.2970]$ and the token ratio in $[0.310,0.313]$. The ratios assume the pass-all log already exists (it took 14,460 agent calls).

Snapshot commits took 873 seconds in total, and the snapshots summed to 9.0 GB; each was deleted after its continuation ran. A snapshot at step 0 averaged 0.94 MB and 1.3 seconds (317 measured), and one after step 0 averaged 52 MB and 2.8 seconds (168 measured).

---

<a id="snapshot-diagnostics"></a>
## Snapshot Diagnostics

### A Snapshot Rule That Does Not Depend on the Gate (Post Hoc)
A rule that commits before every baseline step scoring above $\theta_{\min}$ serves every threshold in $[\theta_{\min},1)$, because the first score above a threshold also exceeds $\theta_{\min}$. Counted from logged scores:
- $\theta_{\min}=0.80$ needs 1,625 snapshots (1.35 per episode, at most 15).
- $\theta_{\min}$ from 0.85 to 0.94 needs 1,548.

Storage and commit time are projected from the 485 measured snapshots by step and exclude restoration. A snapshot at step 0 averaged 0.94 MB and 1.3 seconds (317 measured), and one after step 0 averaged 52 MB and 2.8 seconds (168 measured). Of the 1,625 and 1,548 snapshots, 353 and 317 sit at step 0. The two rules then cost about 67 and 64 GB and about 67 and 64 minutes.

The least set of first-deviation snapshots is smaller (Proposition F.10.1 (`prop:snapshot-design`)). It assumes (T), condition (U), and restoration of each gate at its own first deviations. The set holds the running maxima above $\theta_{\min}$, the steps whose score exceeds $\theta_{\min}$ and every earlier score, that are open for some gate deviating there. On the 1,200 pass-all episodes:
- $\theta_{\min}=0.80$ has 566 running maxima (0.47 per episode, at most 2). None sits at the last step (index 14), so each has a later decision.
- At most 557 are open for the harm indicator, because harm had accrued before the other 9.
- Of the 566, 353 sit at step 0, where restoration needs only the task and seed. So at most 213 need a snapshot (about 11 GB and 10 minutes, projected from the snapshots after step 0).
- The gate $\theta=0.94$ alone needs its 485 first blocks, and the class $[0.94,1)$ needs 520. Both counts include closed first deviations, and the least sets hold at most 479 and 511.

Counting equal scores as running maxima would give 1,547, because the monitor returns exactly 1 at 1,334 of 14,460 steps. An i.i.d. model with the pooled score law of this log, ties included, predicts 0.76 running maxima per episode against 0.47 observed. The prediction uses the observed number of decisions of each episode.

In the replication log (1,000 episodes), running maxima above 0.80 number 0.46 per episode (455 in all), against 1.16 steps above 0.80. The stronger-agent log (1,399 scored episodes) has 0.18 running maxima and 0.24 steps per episode above gate level $\lambda=11.551$ on the doubt scale $s$ (255 and 333 in all).

---

<a id="e7-diagnostics"></a>
## E7 Diagnostics

### Secondary Attribution and Pathwise Details (E7)
- Pathwise (descriptive): a continuation's next output equals the live episode's in 70 of 113 (Gate 1) and 160 of 247 (Gate 2) later-step units with the same observed request, against 0.46 for two live executions that sent an identical request.
- Strata attribution: most of Gate 1's difference lies in later-step continuations (41 of its 60 to 61 net harms; live 0.568, identified 0.527 among 1,020 units), and the rest in agreeing units. If all of Gate 1's difference came from restoration, Lemma `lem:restore` would require an average total-variation error of at least about 0.06 on its continuing prefixes; since a third of the difference lies in agreeing units, which restoration does not touch, the strata do not support attributing all of it to restoration.
- Token metrics: snapshot commits took 3.1 seconds per commit and 61 MB per episode on average across 16,456 snapshots.
- Budgeted strata details: Gate 1, whose harm exceeds the ceiling, certifies 0.30 with 366 of 733 draws.

---

<a id="held-out-retry"></a>
## Held-Out Retry

### Retry Gate on the Held-Out Log
Applied to the held-out log alone, display (5) (`eq:bound`) puts P(harm) in $[0.037,0.493]$ (plug-in outer bounds). Of 300 episodes, 156 agree with the retry gate, 85 are first blocked at step 0, and 59 later. Executing the 85 step-0 continuations once, in fresh sandboxes, narrows the plug-in outer bounds to $[0.040,0.213]$ (95% confidence envelope, exact, 2.5% per side: $[0.021,0.264]$). State restoration passed 0 of 10 checks, so (T) is not justified at later prefixes and the 59 later blocks stay bounded, and the interval still contains 0.10. Run live after the fact (not registered), the gate gave 0.030 (exact 95% $[0.014,0.056]$), 0.010 below the lower end (Newcombe 95% $[-0.042,0.021]$); this earlier held-out protocol was run after the fact, and the evaluation chronology is recorded below.

### Bounds
Additional bounds from the held-out log alone are $[0.420,0.900]$ for P(failure), $[40{,}736,235{,}204]$ for mean tokens, and $[120{,}324,432{,}768]$ for token CVaR at 0.9. After the step-0 continuations, approximate 95% bootstrap outer limits for P(harm) are $[0.020,0.260]$, with $[0.700,0.897]$ for P(failure), $[52{,}504,126{,}022]$ for mean tokens, and $[131{,}648,432{,}768]$ for token CVaR. Graded P(harm) decomposes into agreement mass 0.017, executed contribution 0.003, and bounded contribution $[0.020,0.193]$. Executing the 85 step-0 units eliminated their feasible spread of 0.283. Refusal messages never appear in baseline operation (0 of 4,485 tool messages in the exploratory log).

### Execution Checks
Rerunning the 85 step-0 proposals and monitor outputs reproduced 85 of 85 each, and all 85 context checks passed. For the 59 later blocks, state-restoration checks passed 0 of 10 episodes (threshold 8), with discrepancies in test durations, memory addresses, and tool-call random strings. Their continuations were therefore not executed.

### Live Retry Gate
Run live, the retry gate produced P(failure) 0.857, cap-abort rate 0.153 (46 of 300), mean tokens 65,361, and token CVaR 155,638, with 122 of 300 episodes blocked at least once. Step-0 contributions to total P(harm) were 1/300 in both samples (one harm among 85 executed continuations and one among 74 live blocks; difference 0.000, Newcombe 95% $[-0.016,0.016]$). Never-blocked mass was 0.593 versus 0.520 on the held-out log (difference $+0.073$, $[-0.006,0.151]$). Compared to stopping gate replay from the held-out log, retry changed P(harm) by $-0.007$ $[-0.037,0.023]$, P(failure) by $-0.043$ $[-0.093,0.007]$, and added $+25{,}985$ $[19{,}174,32{,}970]$ mean tokens. Bounds from the exploratory log alone, frozen before the live retry gate completed, were $[0.053,0.400]$. The output of the live retry gate was first read at 05:11 UTC, four minutes before the reading rules were saved at 05:15; the evaluators specified the reading rules without prior access to the output, and no rule or bound changed afterwards (consistent with the run being conducted after the fact, not registered).

---

<a id="log-comparisons"></a>
## Log Comparisons

### The Two Baselines
Baseline P(harm) is 145 of 1,000 on the replication log and 61 of 300 (with one unknown, $[0.203,0.207]$) on the held-out log. The replication log minus the held-out log is $[-0.062,-0.058]$ (Newcombe 95% $[-0.115,-0.011]$). Post hoc, the held-out log's higher replayed abort rate (0.480 versus 0.380 on the replication log) matches its task-mix prediction (0.478 from the replication log's per-task rates).

---

<a id="horizon-diagnostics"></a>
## Horizon Diagnostics

### By Host (Post Hoc)
Splitting the log by host and incident exposure shows consistent trends across cohorts (the companion host table, `tab:d65-host`): stopping replayed P(harm) is constant from $k=15$ to $k=40$, and retry agreement drops primarily within 15 steps. Replayed P(harm) is 0.057 on the incident-free second host and 0.030 on early blocks of the first host. Treating all 39 episodes active during the incident as unknown widens each proportion range to 0.065, raising the stopping gate upper bound to 0.105.

### Host table: Long-Horizon Readings by Host and Incident Exposure

| Cohort | Episodes | Baseline P(harm) | Stopping gate: replayed P(harm) | Stopping gate: share that stops | Retry gate: agreement mass | Retry gate: P(harm) |
|---|---|---|---|---|---|---|
| All episodes (registered) | 600 | 0.168 to 0.197 (0.139, 0.231) | 0.043 to 0.067 (0.028, 0.090) | 0.420 to 0.430 | 0.570 to 0.580 | $[0.043,0.473]$ |
| Second host, no incident | 300 | 0.183 (0.141, 0.232) | 0.057 (0.033, 0.089) | 0.387 | 0.613 | $[0.057,0.423]$ |
| First host, blocks before incident | 200 | 0.165 (0.116, 0.224) | 0.030 (0.011, 0.064) | 0.485 | 0.515 | $[0.030,0.515]$ |
| All episodes, 39 exposed as unknown | 600 | 0.160 to 0.225 (0.132, 0.261) | 0.040 to 0.105 (0.026, 0.132) | 0.400 to 0.465 | 0.535 to 0.600 | $[0.040,0.495]$ |

Original LaTeX source:
```latex
\begin{table}[!ht]
\caption{Long-horizon readings at $k=40$ by host and under incident exposure (post hoc, descriptive). Ranges reflect unknown outcomes, with exact 95\% limits in parentheses; retry P(harm) is a plug-in outer interval.}
\label{tab:d65-host}
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.05}
\begin{tabularx}{\textwidth}{@{}P{0.19\textwidth}P{0.07\textwidth}LLP{0.1\textwidth}P{0.1\textwidth}P{0.11\textwidth}@{}}
\rowcolor{hdrgray}
Cohort & Episodes & Baseline P(harm) & Stopping gate: replayed P(harm) & Stopping gate: share that stops & Retry gate: agreement mass & Retry gate: P(harm) \\
All episodes (registered) & 600 & 0.168 to 0.197 (0.139, 0.231) & 0.043 to 0.067 (0.028, 0.090) & 0.420 to 0.430 & 0.570 to 0.580 & $[0.043,0.473]$ \\
Second host, no incident & 300 & 0.183 (0.141, 0.232) & 0.057 (0.033, 0.089) & 0.387 & 0.613 & $[0.057,0.423]$ \\
First host, blocks before incident & 200 & 0.165 (0.116, 0.224) & 0.030 (0.011, 0.064) & 0.485 & 0.515 & $[0.030,0.515]$ \\
All episodes, 39 exposed as unknown & 600 & 0.160 to 0.225 (0.132, 0.261) & 0.040 to 0.105 (0.026, 0.132) & 0.400 to 0.465 & 0.535 to 0.600 & $[0.040,0.495]$ \\
\end{tabularx}
\end{table}
```

---

<a id="per-horizon-table"></a>
## Per-Horizon Table

This table was moved out of the printed appendix; the paper's pointer to the companion file refers to it.

### Table tab:d65: Long-Horizon Curves from the Long-Horizon Log

**Table tab:d65:** Long-horizon curves from the long-horizon log (600 episodes, 40-step cap). Entries show ranges over unknown outcomes, with exact 95% limits in parentheses; retry P(harm) is a plug-in outer interval. Values from $k=30$ onward equal their $k=30$ values.

| $k$ | Baseline P(harm) | Stopping gate: replayed P(harm) | Stopping gate: share that stops | Retry gate: agreement mass | Retry gate: P(harm) |
|---|---|---|---|---|---|
| 1 | 0.047 to 0.058 (0.031, 0.080) | 0.018 to 0.030 (0.009, 0.047) | 0.222 to 0.232 (0.189, 0.268) | 0.768 to 0.778 (0.732, 0.811) | $[0.018,0.030]$ |
| 5 | 0.152 to 0.167 (0.124, 0.199) | 0.040 to 0.055 (0.026, 0.076) | 0.353 to 0.363 (0.315, 0.403) | 0.637 to 0.647 (0.597, 0.685) | $[0.040,0.373]$ |
| 15 | 0.162 to 0.190 (0.133, 0.224) | 0.043 to 0.067 (0.028, 0.090) | 0.402 to 0.412 (0.362, 0.452) | 0.588 to 0.598 (0.548, 0.638) | $[0.043,0.457]$ |
| 25 | 0.165 to 0.193 (0.136, 0.227) | 0.043 to 0.067 (0.028, 0.090) | 0.417 to 0.427 (0.377, 0.467) | 0.573 to 0.583 (0.533, 0.623) | $[0.043,0.470]$ |
| 40 | 0.168 to 0.197 (0.139, 0.231) | 0.043 to 0.067 (0.028, 0.090) | 0.420 to 0.430 (0.380, 0.471) | 0.570 to 0.580 (0.529, 0.620) | $[0.043,0.473]$ |

Original LaTeX source:
```latex
\begin{table}[!ht]
\caption{Long-horizon curves from the long-horizon log (600 episodes, 40-step cap). Entries show ranges over unknown outcomes, with exact 95\% limits in parentheses; retry P(harm) is a plug-in outer interval. Values from $k=30$ onward equal their $k=30$ values.}
\label{tab:d65}
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.05}
\begin{tabularx}{\textwidth}{@{}P{0.04\textwidth}LLLLP{0.12\textwidth}@{}}
\rowcolor{hdrgray}
$k$ & Baseline P(harm) & Stopping gate: replayed P(harm) & Stopping gate: share that stops & Retry gate: agreement mass & Retry gate: P(harm) \\
1 & 0.047 to 0.058 (0.031, 0.080) & 0.018 to 0.030 (0.009, 0.047) & 0.222 to 0.232 (0.189, 0.268) & 0.768 to 0.778 (0.732, 0.811) & $[0.018,0.030]$ \\
5 & 0.152 to 0.167 (0.124, 0.199) & 0.040 to 0.055 (0.026, 0.076) & 0.353 to 0.363 (0.315, 0.403) & 0.637 to 0.647 (0.597, 0.685) & $[0.040,0.373]$ \\
15 & 0.162 to 0.190 (0.133, 0.224) & 0.043 to 0.067 (0.028, 0.090) & 0.402 to 0.412 (0.362, 0.452) & 0.588 to 0.598 (0.548, 0.638) & $[0.043,0.457]$ \\
25 & 0.165 to 0.193 (0.136, 0.227) & 0.043 to 0.067 (0.028, 0.090) & 0.417 to 0.427 (0.377, 0.467) & 0.573 to 0.583 (0.533, 0.623) & $[0.043,0.470]$ \\
40 & 0.168 to 0.197 (0.139, 0.231) & 0.043 to 0.067 (0.028, 0.090) & 0.420 to 0.430 (0.380, 0.471) & 0.570 to 0.580 (0.529, 0.620) & $[0.043,0.473]$ \\
\end{tabularx}
\end{table}
```

---

<a id="budget-design-table"></a>
## Budget Design Table

This table was moved out of the printed appendix; the paper's pointer to the companion file refers to it.

### Table tab:budget: Budgeted Targeted Execution (Post Hoc, E1 Outcomes)

**Table tab:budget:** **A random sample of continuations certifies the retry gate at under half the cost of live testing (post hoc, E1 outcomes).** Costs are expected marginal costs given the log, failures included; D1's are means over 20,000 random orders plus two Monte Carlo standard errors. P(certify) is given the log and its cohort for the sampling designs and at the live rate 0.030 for the live tests. Over a random log, D1 certifies with probability 0.993 (E1 Check, supplement/appendix-F2.md).

| Design | P(certify) | Executions or episodes | Agent calls | Tokens (millions) |
|---|---|---|---|---|
| D1, target 0.10 (looks 34, 50, 80) | 0.99963 | 46.6 (80th pct. 50) | 387 | 2.56 |
| Live test, target 0.10 | 0.806 | 76 | 854 | 5.30 |
| Execute all 479 | 1 | 479 | 3,960 | 26.12 |
| Fixed 244 executions, target 0.05 | 0.845 | 244 | 2,017 | 13.31 |
| Live test, target 0.05 | 0.800 | 601 | 6,756 | 41.90 |

Original LaTeX source:
```latex
\begin{table}[!ht]
\caption{\textbf{A random sample of continuations certifies the retry gate at under half the cost of live testing (post hoc, E1 outcomes).} Costs are expected marginal costs given the log, failures included; D1's are means over 20,000 random orders plus two Monte Carlo standard errors. P(certify) is given the log and its cohort for the sampling designs and at the live rate 0.030 for the live tests. Over a random log, D1 certifies with probability 0.993 (Appendix~\ref{app:budget-limits}).}
\label{tab:budget}
\centering
\small
\begin{tabular}{@{}lrrrr@{}}
\toprule
Design & P(certify) & Executions or episodes & Agent calls & Tokens (millions) \\
\midrule
D1, target 0.10 (looks 34, 50, 80) & 0.99963 & 46.6 (80th pct.\ 50) & 387 & 2.56 \\
Live test, target 0.10 & 0.806 & 76 & 854 & 5.30 \\
Execute all 479 & 1 & 479 & 3,960 & 26.12 \\
\midrule
Fixed 244 executions, target 0.05 & 0.845 & 244 & 2,017 & 13.31 \\
Live test, target 0.05 & 0.800 & 601 & 6,756 & 41.90 \\
\bottomrule
\end{tabular}
\end{table}
```

---

<a id="assumption-audit"></a>
## Assumption Audit

### Full Assumption Audit: Assumption Audit Across Experimental Demonstrations

The paper does not print this audit; the table below is its archived full version, with its original LaTeX source.

| Assumption | How it is secured | What our runs show | Untested, or fails where |
|---|---|---|---|
| (R) Replay interface | Checkable from logs | Batch-invariant serving reproduced monitor scores on 2,183 of 2,183 repeated inputs (default kernels: 1,686 of 1,811) | Agent reruns on default kernels: step-0 outputs matched in 23 of 99 episodes |
| (A1) Invariance | Controlled by the harness (fresh containers and seeds, a frozen checker); holds as far as serving and tool laws are stable | Replay matched live stopping gates within 0.03 in three settings, an aggregate check | Outputs vary run to run for the stronger agent and the airline; tool or service drift outside the sandbox |
| (A2) No anticipation | Enforced while the gate is unannounced | The unannounced gate stays compatible with replay | Fails for the announced gate: abort rate $+0.070$ |
| (A3) Abort valuation | Declared by the deployer | A blocked call never runs; no retry-gate snapshot listed a process outside the harness | Untested for harm |
| (T) Restoration fidelity | Checked on compared coordinates only | 485 of 485 retry-gate restorations, 3,961 of 3,961 two-gate continuations, and 1,599 of 1,599 relay starts passed their checks | Held-out log: 0 of 10 later-step restorations passed; whole trajectories matched in 302 of 485 |
| (G1) Support | Checkable from the population record law; a finite log cannot show absent support | The pass-all gate cannot emit a refusal (0 of 4,485 tool messages) | Fails for retry refusals; support for the release is unassessed; supported continuation is not exercised |
| (S) Free joint continuations | A property of the model class | Not checkable from logs | Without it, display `eq:bound` gives outer bounds |
| Shared state (Proposition `prop:component-joint`) | Checkable with records that link the agents | Snapshot starts match the relay | Clean starts: only 28 of 1,600 standalone runs left the file tree unchanged; two copies of one agent |
| (H) Independent cohort (Proposition `prop:budget`) | Declared; fresh containers and seeds per episode | Not tested | Budgeted bound shown post hoc on one run; restoration bias excluded |

Original LaTeX source:
```latex
\begin{table}[!ht]
\caption{\textbf{The harness secures (A2) for unannounced gates and controls (A1); (T) is checked on compared coordinates; (A3) and (S) stay declared.} Numbers are from Appendices~\ref{app:demo-gates} to~\ref{app:demo-cues}.}
\label{tab:assumption-audit}
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.08}
\begin{tabularx}{\textwidth}{@{}P{0.15\textwidth}P{0.2\textwidth}LP{0.27\textwidth}@{}}
\rowcolor{hdrgray}
Assumption & How it is secured & What our runs show & Untested, or fails where \\
(R) Replay interface & Checkable from logs & Batch-invariant serving reproduced monitor scores on 2,183 of 2,183 repeated inputs (default kernels: 1,686 of 1,811) & Agent reruns on default kernels: step-0 outputs matched in 23 of 99 episodes \\
(A1) Invariance & Controlled by the harness (fresh containers and seeds, a frozen checker); holds as far as serving and tool laws are stable & Replay matched live stopping gates within 0.03 in three settings, an aggregate check & Outputs vary run to run for the stronger agent and the airline; tool or service drift outside the sandbox \\
(A2) No anticipation & Enforced while the gate is unannounced & The unannounced gate stays compatible with replay & Fails for the announced gate: abort rate $+0.070$ \\
(A3) Abort valuation & Declared by the deployer & A blocked call never runs; no retry-gate snapshot listed a process outside the harness & Untested for harm \\
(T) Restoration fidelity & Checked on compared coordinates only & 485 of 485 retry-gate restorations, 3,961 of 3,961 two-gate continuations, and 1,599 of 1,599 relay starts passed their checks & Held-out log: 0 of 10 later-step restorations passed; whole trajectories matched in 302 of 485 \\
(G1) Support & Checkable from the population record law; a finite log cannot show absent support & The pass-all gate cannot emit a refusal (0 of 4,485 tool messages) & Fails for retry refusals; support for the release is unassessed; supported continuation is not exercised \\
(S) Free joint continuations & A property of the model class & Not checkable from logs & Without it, display~\eqref{eq:bound} gives outer bounds \\
Shared state (Proposition~\ref{prop:component-joint}) & Checkable with records that link the agents & Snapshot starts match the relay & Clean starts: only 28 of 1,600 standalone runs left the file tree unchanged; two copies of one agent \\
(H) Independent cohort (Proposition~\ref{prop:budget}) & Declared; fresh containers and seeds per episode & Not tested & Budgeted bound shown post hoc on one run; restoration bias excluded \\
\end{tabularx}
\end{table}
```

---

## References Cited Only in This File

- Charles F. Manski. 1997. Monotone treatment response. *Econometrica*, 65(6):1311–1334. DOI: [10.2307/2171738](https://doi.org/10.2307/2171738).
