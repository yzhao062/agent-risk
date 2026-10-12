# Appendix H Supplement: Verification Index and Implementation Details

This companion document provides the full verification map, numerical-method disclosures, and execution details for Appendix H (Verification Index).

## 1. Complete Verification Inventory

The table below preserves the complete 15-script mapping between formal claims across the paper and their verification scripts.

| Formal Claim | Script |
|---|---|
| Section 2.3, Appendix A.1, Eq. (3) (`eq:specification`) | `verify_foundation.py` |
| Appendix A.3 (`app:diagnostic`) | `verify_diagnostic.py` |
| Propositions A.4.1 (`prop:replay`), E.1.1 (`prop:replay-cont`), and E.4.3 (`prop:app-identified`); Proposition 3.1 (`thm:logging`); Corollary E.6.1 (`cor:garbling`); injectivity lemma (not printed; archived in supplement/appendix-E.md) | `verify_replay.py` |
| Corollary E.7.1 (`cor:horizon`) | `verify_horizon.py` |
| Proposition A.5.1 (`prop:supported`); Corollary F.6.1 (`cor:model-update`); Lemmas F.3.1 (`lem:p2-truncation`) and F.5.1 (`lem:r4-zero`) | `verify_supported.py` |
| Proposition F.5.2 (`prop:r4-unsupported`) | `verify_unsupported.py` |
| Proposition G.1.1 (`prop:component-joint`); Lemma G.3.2 (`lem:p3-realize`) | `verify_coupling.py` |
| Proposition G.1.1(iv) (`prop:component-joint`); Remark G.4.3 (`rem:p3-eval`) | `verify_eval_awareness.py` |
| Proposition `prop:r4-composition` (not printed; archived in supplement/appendix-G.md) | `verify_composition_replacement.py` |
| Proposition F.7.1(i) and (iii) (`prop:budget`); the budget design table (`tab:budget`; supplement/appendix-D.md#budget-design-table) | `verify_budget.py` |
| Proposition F.8.1 (`prop:budget-limits`); Accept Region and E1 Check (not printed in the paper; supplement/appendix-F2.md); power and call ratios beside the budget design table (`tab:budget`; supplement/appendix-D.md#budget-design-table) | `verify_budget_limits.py` |
| Lemma F.7.2 (`lem:restore`); Remark F.7.3 (`rem:restore`); E1 restoration tolerances (Restoration Allowance in supplement/appendix-D.md; not printed in the paper) | `verify_restoration.py` |
| Proposition F.8.2 (`prop:budget-achievability`); Corollary F.8.3 (`cor:cost-ratio`); Remark F.8.4 (`rem:estimate-log`); E1 Achievability Check (not printed in the paper; supplement/appendix-F2.md) | `verify_budget_sufficiency.py` |
| Propositions F.9.1 (`prop:strata-limits`) and F.9.2 (`prop:strata-achieve`); Remark F.9.3 (`rem:strata-scope`); E7 Check (not printed in the paper; supplement/appendix-F2.md) | `verify_budget_strata.py` |
| Proposition F.10.1 (`prop:snapshot-design`); Remark F.10.2 (`rem:design-evidence`) | `verify_snapshot_design.py` |

## 2. Numerical Methods and Computational Disclosures

The verification scripts rely only on the Python standard library, except that `verify_budget_sufficiency.py` and `verify_budget_strata.py` additionally use `numpy`.

- **Exact Rational Arithmetic**: The scripts compute the laws of their finite mechanisms in exact rational arithmetic using `fractions.Fraction`. Seeded random generation of test cases and rounding for display use floating point.
- **Floating-Point Approximations**: `verify_budget_limits.py` computes logarithms, binomial probabilities, a dynamic program, and a numerical enumeration of the multinomial log with omitted probability mass strictly below $10^{-11}$ in floating point, as stated in its docstring.
- **Dynamic Programs**: `verify_budget_sufficiency.py` uses `numpy` for exact dynamic programs over sequential tests, as documented in its docstring.
- **Finite Verification vs. General Proofs**: Each script constructs finite mechanisms explicitly and computes reported quantities directly from enumerated laws. These calculations provide decisive witnesses for finite examples and counterexamples; Appendices A to G provide general mathematical proofs.

## 3. Execution Commands and Runtimes

Run all scripts from the repository root:

```bash
python checks/verify_foundation.py
python checks/verify_diagnostic.py
python checks/verify_replay.py
python checks/verify_horizon.py
python checks/verify_supported.py
python checks/verify_unsupported.py
python checks/verify_coupling.py
python checks/verify_eval_awareness.py
python checks/verify_composition_replacement.py
python checks/verify_budget.py
python checks/verify_budget_limits.py
python checks/verify_restoration.py
python checks/verify_budget_sufficiency.py
python checks/verify_budget_strata.py
python checks/verify_snapshot_design.py
```

All scripts finish in seconds, except:
- `verify_budget_sufficiency.py` (approximately one minute);
- `verify_unsupported.py` (approximately two minutes);
- `verify_budget_strata.py` (approximately two minutes, including seeded Monte Carlo checks).

For decisive witnesses and worked examples associated with each script, refer to `checks/README.md`.

## 4. Appendix Guide Table

This summary table outlines the contents and roles of each appendix:

| Appendix | What it holds | Supports |
|---|---|---|
| Appendix A (`app:formal`) | The risk specification, the assumptions, and the full statements of the propositions; records that leave a gate change open; Algorithm 1 in symbols and words; symbols | Sections 2 and 3 (`sec:framework`, `sec:results`) |
| Appendix B (`sec:extensions`) | The survey behind the six characteristics, the other characteristics, classical tools, and existing practice | Sections 2.1 and 5 (`sec:characteristics`, `sec:related`) |
| Appendix C (`app:related`) | Extended related work | Section 5 (`sec:related`) |
| Appendix D (`app:demo`) | Setup, run sizes, and the source of every number in Section 4, Tables 2, D.1, D.2, and D.3, and Figures D.1, D.2, and D.3 | Section 4 (`sec:validation`) |
| Appendix E (`app:replay`) | Proofs: deviation replay, logging sufficiency, and long horizons | Sections 3.1 to 3.3 (`sec:replay` to `sec:horizons`) |
| Appendix F (`app:supported`) | Proofs: supported continuation, unsupported inputs, model updates, and budgeted targeted execution | Section 3.4 (`sec:supported`) |
| Appendix G (`app:p3-proof`) | Proofs: component evidence, composition, and oversight cues | Sections 3.5 and 3.6 (`sec:joint`, `sec:cues`) |
| Appendix H (`app:checks`) | Exact-arithmetic scripts that check the finite examples and counterexamples | Appendices A to G (`app:formal` to `app:p3-proof`) |
