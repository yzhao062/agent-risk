# Supplement to Appendix E: Replay, Logging Sufficiency, and Long Horizons

This companion document holds supporting proofs and diagnostic examples for Appendix E of the main paper.

---

## Logging Examples
<a id="logging-examples"></a>

### Status Need Not Be Logged (Original Extended Remark)

The following text provides the unabridged discussion and concrete incident log construction originally presented in Remark E.5.1 (Appendix E.5):

> In check T1-4, a monitor flag $Z$ feeds the gate, the replacement aborts when $Z$ is high with $b_\kappa=\delta_0$, and an incident log records $(Z,Y)$ when the harm $Y$ is $1$ and only "no incident" otherwise.
> The no-incident fibre contains an agreement path with loss $0$ and a terminal path valued $\delta_0$, both with signature $(\delta_0,0)$, and the incident fibres are singletons, so the fibre condition holds although deviation status cannot be recovered from the log. Condition (R) refers to the full record, which contains $Z$; the log is a coarsening of it and need not.

#### Detailed Fibre Structure in Check T1-4
In the finite tree check T1-4:
1. When monitor flag $Z=0$, the replacement rule agrees with the baseline policy ($f^\kappa(I)=f^0(I)$), so the path continues along the agreement set $\mathsf A_\kappa$. If harm $Y=0$, the loss contribution is $C_\kappa = \delta_{\ell(\omega)} = \delta_0$ and continuing indicator $u_\kappa = 0$, giving signature $(\delta_0, 0)$.
2. When monitor flag $Z=1$, the replacement gate triggers an abort ($f^\kappa(I)=\text{abort}$) with declared abort distribution $b_\kappa = \delta_0$. The loss contribution is $C_\kappa = b_\kappa = \delta_0$ and continuing indicator $u_\kappa = 0$, also yielding signature $(\delta_0, 0)$.
3. Because both paths yield the identical signature $(\delta_0, 0)$, pooling them into a single coarse log value ("no incident" when $Y=0$) satisfies the fibre condition of Proposition 3.1: $(C_\kappa, u_\kappa)$ is constant on the fibre $g^{-1}(\text{"no incident"})$.
4. Consequently, the evaluator can compute the exact replay summary $s_\kappa(P)$ from the coarse log $g_\# P$ even though the log cannot distinguish whether an episode agreed with the baseline or was aborted by the gate.
5. In contrast, Condition (R) governs the full execution record, which contains $Z$ and distinguishes all branches. The log $g$ is a coarsening of the full record and is not required to record the internal decision status.

---

## Expanded Proofs
<a id="expanded-proofs"></a>

### Proof of Proposition A.4.1 (Deviation Replay, Part i)

The unabridged proof of Proposition A.4.1(i) with expanded summation:

```latex
\begin{proof}[Proof of Proposition~\ref{prop:replay}(i)]
Let $B\subseteq\lossset$.
By Lemma~\ref{lem:app-prefix}(a),
\begin{align*}
P^\kappa(L\in B,\mathsf A_\kappa,\Omega_r)
&=\textstyle\sum_{\omega\in\mathsf A_\kappa\cap\Omega_r,\,\ell(\omega)\in B}P^\kappa(\omega)\\
&=\textstyle\sum_{\omega\in\mathsf A_\kappa\cap\Omega_r,\,\ell(\omega)\in B}P^0(\omega)
=P^0(L\in B,\mathsf A_\kappa,\Omega_r).
\end{align*}
Dividing by $P^\kappa(\Omega_r)=P^0(\Omega_r)>0$, from Lemma~\ref{lem:app-prefix}(b), gives (i).
The argument compares two path laws on one finite set and never forms a joint law of baseline and replaced trajectories.
\end{proof}
```

### Proof of Proposition A.4.1 (Deviation Replay, Part ii)

The unabridged proof of Proposition A.4.1(ii) with explicit cylinder calculations:

```latex
\begin{proof}[Proof of Proposition~\ref{prop:replay}(ii)]
Fix $h\in\mathcal D^{\mathrm{term}}_\kappa(r)$.
If $P^0([h])>0$, Lemma~\ref{lem:app-prefix}(b) and (A3) give
$P^\kappa(L\in B,[h])=P^\kappa([h])\,P^\kappa(L\in B\mid[h])=P^0([h])\,b_\kappa(B\mid h^+)$.
If $P^0([h])=0$, both sides vanish by Lemma~\ref{lem:app-prefix}(b).
Summing over $\mathcal D^{\mathrm{term}}_\kappa(r)$ and dividing by $P^0(\Omega_r)$ gives
\begin{equation*}
P^\kappa(L\in B,\sigma_\kappa\le T,\text{terminal}\mid r)
=\sum_{h\in\mathcal D^{\mathrm{term}}_\kappa(r)}P^0([h]\mid r)\,b_\kappa(B\mid h^+).
\end{equation*}
The right side is the integral in (ii), because $b_\kappa(B\mid h^+_\kappa(\tau))$ equals $b_\kappa(B\mid h^+)$ for every $\tau\in[h]$.
The baseline probability of an abort output never enters.
\end{proof}
```

### Proof of Lemma E.4.2 (Unrestricted Class)

The complete proof sketch establishing simultaneous attainment across all continuing prefixes:

```latex
\begin{proof}[Proof sketch]
For each continuing prefix $h$, select replacement-consistent paths that achieve each outcome in $\mathcal X_h$ and form a mixture matching $Q_h$, while using the realized extensions for terminal prefixes.
Because unreplaced kernels read the full history, their transition probabilities can be set to the baseline transitions on the support of $P$ and to the target continuation transitions on cylinders extending each $h^+$.
These cases do not conflict because baseline paths in $\Omega^0$ never encounter a replayed prefix $h^+$.
The product path law therefore recovers baseline law $P$ on $\Omega^0$ and simultaneously realizes the target conditional laws $Q_h$ after replacement.
\end{proof}
```

### Proof of Proposition E.4.3 (Identified Set)

The complete derivation of the identified set and sharpness under saturation:

```latex
\begin{proof}
Let $M\in\mathcal M(P)$.
By \eqref{eq:app-partition}, $\phi_\#P^\kappa_M(L\in\cdot\mid r)$ is a sum of an agreement part, a terminal part, and a continuing part.
The first two parts equal the two terms of $\nu$ by parts (i) and (ii) [R, A1, A2, A3].
For $h\in\mathcal D^{\mathrm{cont}}_\kappa(r)$, Lemma~\ref{lem:app-prefix}(b) gives $P^\kappa_M([h]\mid r)=P([h]\mid r)$, which vanishes unless $h\in\mathcal D^{\mathrm{cont}}_\kappa(r,P)$.
Under $P^\kappa_M$ the decision kernel is $\delta_{f^\kappa}$ at $h$ and at every later decision [R].
Every path of positive probability in $[h]$ is therefore a replacement-consistent extension of $h^+$, and $Q^M_h=\phi_\#P^\kappa_M(L\in\cdot\mid[h])$ lies in $\Delta(\mathcal X_h)$.
Hence $\phi_\#P^\kappa_M(L\in\cdot\mid r)=\nu+\sum_hw_hQ^M_h\in\mathcal S_\phi(P)$.
If $\mathcal M$ is saturated at $P$, Definition~\ref{def:app-saturated} attains every element of $\mathcal S_\phi(P)$.
\end{proof}
```

### Proof of Proposition E.4.4 (Order Bounds)

The full proof sketch for stochastic dominance and extreme attainment:

```latex
\begin{proof}[Proof sketch]
For (a), placing all continuation mass at $\min\mathcal X_h$ and $\max\mathcal X_h$ bounds each continuation CDF pointwise, yielding first-order stochastic dominance \citep{shaked2007stochastic}.
For (b), monotonicity in stochastic dominance transfers these extremal distributions to bounds on monotone functionals, with attainment at $\eta^\pm$ \citep{manski2003partial}.
For (c), convexity of $\mathcal S_\phi(P)$ ensures that mixtures between $\eta^-$ and $\eta^+$ belong to the set.
The intermediate value theorem completes the claim because the mean and exceedance probabilities are affine in the mixture parameter, while CVaR is continuous as a finite minimum of affine functions \citep{rockafellar2002cvar}.
\end{proof}
```

### Proof of Proposition E.1.1 (Continuing Deviations)

The complete deduction combining saturation and order bounds:

```latex
\begin{proof}[Proof of Proposition~\ref{prop:replay-cont}]
With every coordinate recorded, the evidence is $P^0$. Condition (S) makes the class saturated at $P^0$ (Definition~\ref{def:app-saturated}), so Proposition~\ref{prop:app-identified} gives the identified set with equality; its $\nu$ is the measure of the statement, and $\sum_hw_hQ_h=\int Q_h\,P^0(dh,\text{continuing}\mid r)$. Proposition~\ref{prop:app-order} gives the extremes, the intermediate values, and the finite range of VaR. Without (S), the inclusion of Proposition~\ref{prop:app-identified} makes these outer bounds.
\end{proof}
```

### Injectivity (archived here; not printed in the paper; checked by checks/verify_replay.py)

The full proof sketch showing that the identified set uniquely isolates the summary signature:

```latex
\begin{proof}[Proof sketch]
Under full continuing support $\mathcal X_h=\lossset$, the identified set takes the form $\{\nu_\kappa+\varepsilon_\kappa Q: Q\in\Delta(\lossset)\}$.
Because $|\lossset|\ge2$, taking the pointwise minimum across this set at each outcome isolates the subprobability mass $\nu_\kappa(\{l\})$, which in turn determines $\varepsilon_\kappa=1-\nu_\kappa(\lossset)$.
Hence the identified set uniquely determines the summary signature $s_{\kappa,r}(P)=(\nu_\kappa,\varepsilon_\kappa)$.
Logging map preservation then follows from Proposition~\ref{thm:logging} by combining the sufficiency of constant signatures with point-mass witness laws for necessity.
\end{proof}
```

### Proof of Corollary E.6.1 (Garbling)

The Blackwell experiment comparison argument:

```latex
\begin{proof}[Proof sketch]
Let $\psi(M)$ be the query and $M_0\in\mathcal M$ the true mechanism.
Because $g'=G\circ g$, any mechanism matching the baseline law under $g$ also matches under $g'$, so the compatible class under the finer record is a subset of that under the coarser record \citep{blackwell1951comparison,blackwell1953equivalent}.
Applying $\psi$ to both compatible classes yields the set inclusion.
\end{proof}
```

### Proof of Corollary E.7.1 (Long Horizons)

The full chain-rule and hazard product argument:

```latex
\begin{proof}[Proof sketch]
For (i), the agreement event decomposes by the chain rule across survival steps, with each hazard identified from recorded baseline inputs via Lemma~\ref{lem:app-prefix}(b).
For (ii), when no deviations continue, agreement and declared abort mass sum to one, so constancy of the signature identifies the full replacement law via Proposition~\ref{thm:logging}.
For (iii), the identified mean interval endpoints from Proposition~\ref{prop:app-order} yield a width equal to the spread integrated against continuing mass.
When every deviation continues, this continuing mass equals one minus the agreement mass from (i).
\end{proof}
```

---

## Finite Setting Reference
<a id="finite-setting-reference"></a>

The complete baseline setting from Appendix E.2:
- **Paths and Trees**: $\Omega$ is a finite set of complete paths $\omega=(v_1,\dots,v_{n(\omega)})$ listing execution, observation, and post-abort coordinates in declared generation order. No path is a proper prefix of another. Histories $\mathcal H$ are proper prefixes, including the empty history.
- **Kernels**: $\mathcal N$ is a finite set of kernel names. Declared order assigns kernel name $\mathrm{nm}(h)\in\mathcal N$ and declared input $\pi(h)$ to history $h$.
- **Path Law**: $P_M(\omega)=\prod_{j=0}^{n(\omega)-1}k^M_{\mathrm{nm}(\omega_{1:j})}\bigl(v_{j+1}\mid\pi(\omega_{1:j})\bigr)$.
- **Cylinder Probability**: $P_M([h])=\prod_{j=0}^{|h|-1}k^M_{\mathrm{nm}(h_{1:j})}\bigl(h_{j+1}\mid\pi(h_{1:j})\bigr)$.
- **Partition**: $\Omega=\mathsf A_\kappa\sqcup\bigsqcup_{h\in\mathcal D_\kappa}[h]$, where $\mathsf A_\kappa=\{\sigma_\kappa=\infty\}$ is the agreement set and $\mathcal D_\kappa$ is the set of first-deviation histories.
