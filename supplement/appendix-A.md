# Appendix A Companion: Specification, Formal Statements, and Evaluation Procedure

This companion file preserves full worked examples, procedural descriptions in words, and complete notation tables moved from Appendix A of the main paper ("agent-risk: Risk Assessment for Changes to AI Agents") to maintain a compact PDF footprint.

---

## Diagnostic Examples
<a id="diagnostic-examples"></a>

### Omitted Harm Timing under a Stricter Gate: Complete Worked Construction

Below is the complete worked construction from Appendix A.3 demonstrating how omitting harm timing from logs leaves the risk of a stricter gate under an accrued-harm abort valuation partially identified on the unit interval $[0,1]$.

One episode has steps $t=0,1,2,3$, and the conditioning record is trivial ($t=-1$). At every step the agent proposes one fixed call $u$, the monitor emits the score $m_t=\mathbf 1\{t=2\}$, and the baseline gate executes the call. An environment variable $Z\in\{\mathrm{early},\mathrm{late}\}$, never recorded, has $P(Z=\mathrm{early})=p$. The sandbox reads $t$, $Z$, and the executed call, and it sets the step harm $h_t=1$ if $(Z,t)$ is $(\mathrm{early},1)$ or $(\mathrm{late},3)$, and $h_t=0$ otherwise. Over executed steps, the loss $L=\max_th_t$ equals $1$ on both baseline paths.

Logging retains the proposals, the scores, the gate decisions, and $L$, but not $h_t$, so every $p\in[0,1]$ gives the same record law, a point mass. At threshold $\theta\in[0,1)$, the stopping gate $\kappa_\theta$ reads $I_t=(u_t,m_t)$, which the record contains, so (R) holds; it changes only the gate, so (A1) and (A2) hold. Its first deviation is terminal at $\sigma_\kappa=2$ on both paths, and the declared valuation $b_\kappa(\cdot\mid h^+)=\delta_{\max(h_0,h_1)}$ keeps the harm accrued before the abort and adds none. Proposition A.4.1(ii) then gives $P^\kappa(L=1)=p$: the gate keeps the harm at step $1$ and removes the harm that would come at step $3$.

Take the family that fixes every kernel except the law of $Z$. Every member satisfies (R) and (A1) to (A3) and has the observed record law, so the identified set of $P^\kappa(L=1)$ is $[0,1]$ however many episodes are logged. The two baseline paths share one log value but have contributions $\delta_1$ and $\delta_0$, so the fibre condition of Proposition 3.1 fails. Adding $h_t$ to the record makes the contribution a function of the record and identifies $P^\kappa(L=1)=p$; the coarser record's set $[0,1]$ contains $\{p\}$, as Corollary E.6.1 requires.

---

## Algorithm in Words
<a id="algorithm-in-words"></a>

### Line-by-Line Procedural Description of Algorithm 1

Below is the complete text of the procedure from Appendix A.6 stating each line of Algorithm 1 in words, preserving the identical control flow and references.

- **Input:** specification $\mathfrak R$, model family $\mathcal M_0$, records $r$, replacement $\kappa$, query $\rho\circ\phi$, decision rule.
- **Output:** identification status, an estimate or interval, and the evidence still missing.

1. **Line 1:** Declare replaced kernels, inputs, abort valuation $b_\kappa$ of (A3), and the structural assumptions (A1), (A2), and restoration fidelity (T) linking evidence to the target.
2. **Lines 2 to 3:** If (A1) or (A2) is not established and the additional changes are not explicitly modeled, return that this analysis does not evaluate the target.
3. **Line 4:** Where the replay interface (R) and records permit, recover contributions and deviation labels $\sigma_\kappa$, replayed prefix $h^+_\kappa$, and deviation events; labels need not be individually observable.
4. **Lines 5 to 6:** If under the stated law class, the log passes the logging-sufficiency check (Proposition 3.1) and retains feasible-outcome information $\mathcal X_h$, evaluate the identified sum of agreement and terminal contributions from $P^0(\cdot\mid r)$ and $b_\kappa$.
5. **Line 7:** For all parts left unresolved, including those of a failed check:
   - **Lines 8 to 9:** If (G1) to (G4) hold, apply the sequential g-formula (Proposition A.5.1).
   - **Lines 10 to 11:** Else if snapshots at the replayed prefixes $h^+_\kappa$ are kept and restoration fidelity (T) is justified, restore each prefix and execute only its continuation, or, for binary harm under hypothesis (H), a uniform random sample of the continuations with the bound of Proposition F.7.1.
   - **Lines 12 to 13:** Else optimize over compatible models and latent record completions ($\Theta^\kappa_t(r)$ of Display (1), the set $\Theta(\mathcal M)$ of Appendix A.2; Proposition A.5.1); with full records and a risk functional nondecreasing in first-order stochastic dominance, this is the interval of Display (5); label the result sharp or outer.
6. **Lines 14 to 15:** If the identification status and sampling uncertainty leave the decision unresolved, distinguish additional episodes, richer records, justified assumptions, and new execution.
7. **Line 16:** Return the identification status (identified, bounded, or needs execution) and the decision with sampling uncertainty.

---

## Symbols
<a id="symbols"></a>

### Symbol reference: Symbols of Sections 2 and 3 and Appendix A

Below is the complete reference table mapping all mathematical symbols of Sections 2 and 3 and Appendix A to their informal and exact definitions.

| Symbol | Meaning | Exact Definition |
| :--- | :--- | :--- |
| $\mathfrak R=(E,\ell,K,\mathcal C)$ | Risk specification: execution mechanism, consequence map, observation mechanism, admissible replacements | Appendix A.1 (`app:formal-spec`) |
| $T$; $t$ | Finite horizon; step index, shared by execution and records | Appendix A.1 (`app:formal-spec`) |
| $u_t$; $m_t$ | Tool calls proposed at step $t$; the monitor's score for them | Section 2.2 (`sec:running`) |
| $\kappa$; $\kappa_\theta$; $\theta$ | A replacement (a change); the stopping gate; its score threshold | Appendix A.1 (`app:formal-spec`) |
| $L=\ell(\omega)$; $\mathbb L$ | Full-episode loss vector (failure, harm, cost); its finite set of values | Appendix A.1 (`app:formal-spec`) |
| $R_t=(O_0,\ldots,O_t)$; $r$ | Evaluator records up to $t$; a value of $R_t$ with positive baseline probability | Appendix A.1 (`app:formal-spec`) |
| $\Omega$; $P_{E,K}$; $P^0$; $P^\kappa$ | Complete joint paths; their law; the baseline law; the law under $\kappa$ | Appendices A.1, A.2 (`app:formal-spec`, `app:formal-question`) |
| $I_t$; $f^0_t$; $f^\kappa_t$ | Declared input of the replaced decision kernel; the baseline and replacement rules | Appendix A.1 (`app:formal-spec`) |
| $Q^\kappa_t(\cdot\mid r)$ | Loss law of the changed system given the record at the change | Appendix A.2 (`app:formal-question`) |
| $q^\kappa_t(r)$; $\Theta^\kappa_t(r)$ | Risk of the change: $\rho$ applied to the law of $\phi(L)$ under $Q^\kappa_t(\cdot\mid r)$; its values under the models that reproduce the records | Display (1) (`eq:risk`) |
| $\phi$; $\phi_\#Q$; $\rho$ | Coordinate or aggregation of $L$; the law of $\phi(L)$ under $Q$; risk functional (probability, mean, CVaR) | Appendix A.2 (`app:formal-question`) |
| $\mathcal M_0$; $\mathcal M$; $\psi$; $\Theta(\mathcal M)$ | Declared model family; models that reproduce the records; a query; the identified set | Appendix A.2 (`app:formal-question`) |
| (A1); (A2) | Invariance; prefix preservation and no anticipation | Appendix A.2 (`app:formal-question`) |
| (T) | Restoration fidelity of a continuation run from a restored prefix | Appendix A.2 (`app:formal-question`) |
| $\Omega^0$ | Paths on which every replaced decision equals the baseline rule | Appendix A.4 (`app:formal-replay`) |
| $\sigma_\kappa$; $\mathsf A_\kappa$ | First deviation of the new rule from the logged one; the agreement set $\{\sigma_\kappa>T\}$ | Appendix A.4 (`app:formal-replay`) |
| $h^+_\kappa$ | Replayed prefix: the history before the deviation followed by the replacement output | Appendix A.4 (`app:formal-replay`) |
| terminal; continuing | A deviation whose replacement output is an abort; any other deviation | Appendix A.4 (`app:formal-replay`) |
| (R); (A3); $b_\kappa$ | Replay interface; declared abort valuation; the declared loss law after a stop | Appendix A.4 (`app:formal-replay`) |
| agree; stop; continue | The events $\mathsf A_\kappa$, $\{\sigma_\kappa\le T,\ \text{terminal}\}$, and $\{\sigma_\kappa\le T,\ \text{continuing}\}$ | Display (2) (`eq:split`) |
| $E^0[\,\cdot\mid r]$ | Expectation of a path function under $P^0(\tau\mid r)$ | Display (2) (`eq:split`) |
| $\mathcal X_h$; (S) | Feasible values of $X=\phi(L)$ after a continuing prefix $h$; saturated continuation | Section 3; Appendix E.1 (`sec:results`, `app:replay-cont`) |
| $\nu^-$; $\nu^+$ | Law of $X$ given $r$ when agreeing episodes keep their logged value, stopped episodes follow the declared valuation, and each continuing episode takes the least (greatest) value in $\mathcal X_h$ | Display (5) (`eq:bound`) |
| $C_\kappa$; $u_\kappa$ | An episode's combined contribution to the first two terms of (2); its missing mass | Proposition 3.1 (`thm:logging`) |
| $\mathcal P_r$; $s_{\kappa,r}$ | Baseline laws on $\Omega^0$ that charge $r$; the replay summary of a law | Proposition 3.1 (`thm:logging`) |
| $g$; $g_\#P$; $\Psi$ | Logging map; the law of the log; a map from the log law to the summary | Proposition 3.1 (`thm:logging`) |
| $\eta_j(r)$ | Deviation hazard at step $j$ among episodes that agree before $j$ | Corollary E.7.1 (`cor:horizon`) |
| (G1) to (G4) | Support, recorded kernel inputs, recorded loss coordinates, recorded replacement inputs | Appendix A.5 (`app:formal-supported`) |
| $S$; $P_S$; $U_i$ | Shared state; its law; composed states never seen by component $i$ alone | Section 3.5 (`sec:joint`) |
