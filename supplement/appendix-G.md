# Appendix G Supplement: Component Evidence and Composition

This companion document preserves the complete worked examples, detailed constructions, and expanded arguments moved from Appendix G of *agent-risk: Risk Assessment for Changes to AI Agents*.

---

<a id="dependence-examples"></a>
## Dependence Examples

This section preserves the three canonical failure modes of conditional independence (CI) given shared state $S$, as verified in `checks/verify_coupling.py`.

### 1. Pairwise Conditional Independence Without Mutual Independence

Mutual conditional independence requires that the joint conditional distribution factors completely:
$$P(L_1=l_1, \dots, L_n=l_n \mid S=s) = \prod_{i=1}^n F_i(l_i \mid s).$$
Pairwise conditional independence between every pair of components does not imply mutual conditional independence when $n \ge 3$.

**Construction.**
Let $n=3$ with shared state $S=0$ constant. Let $L_1$ and $L_2$ be independent fair binary losses drawn from $\text{Bernoulli}(1/2)$. Define the third component's loss by the parity:
$$L_3 = L_1 \oplus L_2.$$
For every pair $(i, j) \in \{(1,2), (1,3), (2,3)\}$ and any $(a, b) \in \{0, 1\}^2$:
$$P(L_i=a, L_j=b) = \frac{1}{4} = P(L_i=a) P(L_j=b).$$
Thus, all pairs are conditionally independent given $S$. However, the product rule misstates the joint probability of concurrent failure:
$$\prod_{i=1}^3 P(L_i=1 \mid S=0) = \left(\frac{1}{2}\right)^3 = \frac{1}{8},$$
whereas the actual joint probability of all three failing simultaneously is:
$$P(L_1=1, L_2=1, L_3=1 \mid S=0) = P(L_1=1, L_2=1, 1 \oplus 1 = 1) = 0.$$
Paired records $e_{\mathrm{pair}} = (Q^{(1)}, Q^{(2)}, Q^{(3)})$ observe only pairwise distributions with $S$ and cannot detect this high-order parity dependence.

### 2. Direct Handoff Absent From Shared State

When one component reads the output of an earlier component directly without that dependency passing through the declared shared state $S$, conditional independence given $S$ fails.

**Construction.**
Let $n=2$ with constant state $S=0$. Component 1 draws a private fair bit $L_1 \sim \text{Bernoulli}(1/2)$. Component 2 directly copies component 1's action:
$$L_2 = L_1.$$
The marginal laws are $F_1(1 \mid 0) = 1/2$ and $F_2(1 \mid 0) = 1/2$. The product formula predicts joint failure probability:
$$P^{\otimes}(L_1=1, L_2=1 \mid S=0) = F_1(1 \mid 0) F_2(1 \mid 0) = \frac{1}{2} \times \frac{1}{2} = \frac{1}{4}.$$
The true joint probability is:
$$P(L_1=1, L_2=1 \mid S=0) = P(L_1=1) = \frac{1}{2}.$$
Because the direct handoff bypassed $S$, paired records cannot establish the joint failure probability.

### 3. Common Effect (Collider Conditioning on Budget)

When two components act independently, but the shared state $S$ records an outcome or resource accumulation determined by both components, conditioning on $S$ introduces spurious dependence.

**Construction.**
Let $L_1, L_2 \sim \text{Bernoulli}(1/2)$ be independent private component losses. Let the shared state record the total budget spent across the episode:
$$S = L_1 + L_2 \in \{0, 1, 2\}.$$
Unconditionally, $L_1$ and $L_2$ are independent: $P(L_1=l_1, L_2=l_2) = 1/4$ for all $(l_1, l_2) \in \{0, 1\}^2$.
Now condition on state $S=1$:
$$P(L_1=1, L_2=1 \mid S=1) = 0,$$
since $L_1=1$ and $L_2=1$ implies $S=2$. However, the conditional marginals are:
$$F_1(1 \mid S=1) = P(L_1=1 \mid L_1 + L_2 = 1) = \frac{1}{2},$$
$$F_2(1 \mid S=1) = P(L_2=1 \mid L_1 + L_2 = 1) = \frac{1}{2}.$$
The product formula yields:
$$F_1(1 \mid S=1) F_2(1 \mid S=1) = \frac{1}{2} \times \frac{1}{2} = \frac{1}{4} \neq 0.$$
Conditioning on a common effect (the shared budget spend) induces negative dependence between components.

### 4. Shared Outage and Conditional Fréchet Bounds

Two booking paths share one provider. The logged flag $S$ has $P_S(0)=3/4$ and $P_S(1)=1/4$. An unlogged outage fails both paths with probability $1/2$ when $S=1$, and private faults have probabilities $1/10$ and $1/5$. Both paths fail with probability $39/400$ under the product rule, $11/80$ under the shared outage, and anywhere in $[1/20,9/40]$ without a dependence restriction; linked records separate the two mechanisms. These are the finite witnesses in `checks/verify_coupling.py`.

---

<a id="timing-examples"></a>
## Timing Examples

This section preserves the complete worked counterexample showing why condition (T$_i$) ("State fixed first") cannot be omitted in Proposition G.5.3 (`prop:r4-composition`, archived below; not printed in the paper) and Remark G.5.4(c) (`rem:r4-standalone-inv`, archived below).

### Necessity of Condition (T$_i$)

Condition (T$_i$) requires that the shared state $S$ is fixed before component $i$ acts: specifically, $S$ must be a function of coordinates $V_{\mathcal W_i}$ preceding component $i$'s internal block $\mathsf C_i$. When a shared state variable is written after component $i$ acts, it can correlate with component $i$'s internal actions, invalidating the transport of standalone evidence even when block order, interface reading, and interface support all hold.

**Construction.**
Consider a system with $n=2$ components. Let component 2 have:
- Constant interface state $q_2 = 0$.
- A private internal fair coin flip $X \sim \text{Bernoulli}(1/2)$.
- Component loss $L_2 = X$.

Define the shared state as the pair $S = (q_1, q_2)$. In this system, coordinate $q_1$ is written after component 2 generates $X$:
1. In standalone deployments $E^0_{\mathrm{joint}}$, $q_1$ is generated independently as a fair coin flip: $q_1 \sim \text{Bernoulli}(1/2)$.
   The standalone state distribution has support $\mathcal S^{\mathrm{sa}}_2 = \{(0,0), (1,0)\}$, each with probability $1/2$.
   The standalone conditional loss laws are:
   $$Q^{(2)}(L_2 = 1 \mid S = (0,0)) = \frac{1}{2}, \qquad Q^{(2)}(L_2 = 1 \mid S = (1,0)) = \frac{1}{2}.$$
2. In the composed deployment under $\kappa_{\mathrm{conn}}$, the state kernel draws an independent fair bit $B \sim \text{Bernoulli}(1/2)$ and writes:
   $$q_1 = X \wedge B.$$
   Because $q_1$ reads $X$, the state coordinate $q_1$ is generated downstream of component 2's action.

**Verification of Conditions:**
- (Blk$_2$) holds because component 2's interface coordinate $q_2$ precedes its internal block $\mathsf C_2$.
- (Rd$_2$) holds with $\psi_2(s) = s_2 = 0$, so $V_{\Sigma_2} = \psi_2(S)$ almost surely. Note that $\psi_2$ is not injective on the state support $\mathcal S_+ = \{(0,0), (1,0)\}$.
- (Supp$^{\Sigma}_2$) holds because the interface value $S^{(2)} = 0$ occurs in both standalone and composed executions with probability 1.
- (T$_2$) fails because $S$ depends on $q_1 = X \wedge B$, which involves the internal coordinate $X \in \mathsf C_2$ and is not a function of $V_{\mathcal W_2}$.

**Resulting Failure of Invariance:**
In the composed execution:
- $P(S = (1,0)) = P(X=1, B=1) = 1/4$.
- $P(S = (0,0)) = 1 - 1/4 = 3/4$.
The conditional laws of $L_2$ given $S$ in the composition are:
$$P(L_2 = 1 \mid S = (1,0)) = \frac{P(X=1, X=1, B=1)}{P(S = (1,0))} = \frac{1/4}{1/4} = 1.$$
$$P(L_2 = 1 \mid S = (0,0)) = \frac{P(X=1, B=0)}{P(S = (0,0))} = \frac{1/4}{3/4} = \frac{1}{3}.$$
However, the intrinsic component block kernel from the block kernel definition (display `eq:r4-block-kernel`, archived below) gives:
$$G_2(1 \mid 0) = \frac{1}{2}.$$
Under the joint-baseline declaration $\varsigma^{\mathrm{sa}}_2 = \varsigma$, both standalone and composed deployments share the state support $\{(0,0), (1,0)\}$, so support condition (Supp$_2$) holds. Yet:
$$Q^{(2)}(L_2 = 1 \mid S = s) = \frac{1}{2} \neq P(L_2 = 1 \mid S = s)$$
at both states $s \in \{(0,0), (1,0)\}$. Invariance condition (Inv$_2$) fails.
This counterexample shows why (T$_i$) cannot be dropped from Proposition G.5.3(i) (`prop:r4-composition`, archived below; not printed in the paper); Remark G.5.4(d) (`rem:r4-standalone-inv`, archived below) gives another route through recorded fields.

---

<a id="specification-recap"></a>
## Full Specification Recap

For complete reference, the full formal joint specification structure $\mathfrak R_{\mathrm{joint}}$ from Section 2 is:
$$\mathfrak R_{\mathrm{joint}}=(E_{\mathrm{joint}},\ell_{\mathrm{joint}},K_{\mathrm{joint}},\mathcal C_{\mathrm{joint}}),$$
where:
1. $E_{\mathrm{joint}} = (E^0_{\mathrm{joint}})^{\kappa_{\mathrm{conn}}}$ is the composed execution mechanism obtained by applying connection replacement $\kappa_{\mathrm{conn}}$ to independent standalone baseline $E^0_{\mathrm{joint}}$.
2. $\ell_{\mathrm{joint}} = h \circ (S, \mathbf L)$ is the joint loss map defined from shared state $S$ and component losses $\mathbf L = (L_1, \dots, L_n)$.
3. $K_{\mathrm{joint}}$ is the observation mechanism recording coordinates $V_{\mathcal O}$ with $\mathcal O = \bigsqcup_i O_i$.
4. $\mathcal C_{\mathrm{joint}} = \{\mathrm{id}, \kappa_{\mathrm{conn}}\}$ is the intervention family containing the baseline and the connection replacement.


<a id="composition-replacement-archive"></a>
## Archived Composition as a Replacement

The complete former Section G.5 is preserved below as LaTeX source. Its structural sufficient conditions are additional to the conditional-law invariance assumed directly in Section 3.5; Proposition G.1.1(iv) in the manuscript no longer asserts these conditions or the recorded-output exception. References to the archived result labels in the verification scripts and this companion document refer to this source.

```latex
\subsection{Composition as a Replacement}
\label{app:r4-composition-replacement}

Connecting agents through shared memory is an admissible replacement $\kappa_{\mathrm{conn}}$ of a joint specification whose baseline is standalone runs \citep{robins1986new,pearl1995sequential,pearl2011transportability,manski2003partial}.

Components $i\in\{1,\dots,n\}$ have standalone laws $P^{\mathrm{sa}}_i$.
Coordinates $V_1,\dots,V_N$ are partitioned into $A_i$ ($[N]=\bigsqcup_i A_i$), with outside interface $\Sigma_i\subseteq A_i$ and internal block $\mathsf C_i=A_i\setminus\Sigma_i$; $\Sigma=\bigsqcup_i\Sigma_i$ and $\mathsf C=\bigsqcup_i\mathsf C_i$.
Baseline $E^0_{\mathrm{joint}}$ runs standalone deployments independently with law
\begin{equation}
P^0(v)=\prod_{j=1}^{N}k_j(v_j\mid v_{S_j}),\qquad v\in\textstyle\prod_{j}\mathcal V_j .
\label{eq:r4-joint-base}
\end{equation}
Deployment $i$ logs $O_i\subseteq A_i$, giving record $V_{\mathcal O}$ with $\mathcal O=\bigsqcup_iO_i$.
Replacement $\kappa_{\mathrm{conn}}$ replaces state kernels at $D_\kappa\subseteq\Sigma$ with scheduler and shared-state kernels $k^\kappa_j(\cdot\mid v_{S^\kappa_j})$ for $S^\kappa_j\subseteq[j-1]$; kernels outside $D_\kappa$ and blocks $\mathsf C_i$ remain unchanged, yielding
\begin{equation}
P^\kappa(v)=\prod_{j\in D_\kappa}k^\kappa_j(v_j\mid v_{S^\kappa_j})\prod_{j\notin D_\kappa}k_j(v_j\mid v_{S_j}).
\label{eq:r4-conn}
\end{equation}
Shared state is $S=\varsigma(V_{\Sigma^S})$ ($\Sigma^S\subseteq\Sigma$) with law $P_S$ and support $\mathcal S_+$.
Output set $Y_i\subseteq\mathsf C_i$ holds coordinates read by losses or downstream kernels.
Losses are $L_i=\ell_i(V_{\Sigma_i},V_{Y_i})$ and $L=h(S,\mathbf L)$.
Interface value $S^{(i)}=V_{\Sigma_i}$ has standalone support $\mathcal U^{\mathrm{sa}}_i=\{u:P^{\mathrm{sa}}_i(S^{(i)}=u)>0\}$.
Block and loss kernels are identical under both mechanisms:
\begin{equation}
k^{\mathrm{blk}}_i(y\mid u)=\sum_{c\in\mathcal V_{\mathsf C_i}:\,c_{Y_i}=y}\ \prod_{j\in\mathsf C_i}k_j\bigl(c_j\mid(u,c)_{S_j}\bigr),\qquad
G_i(l\mid u)=\sum_{y:\,\ell_i(u,y)=l}k^{\mathrm{blk}}_i(y\mid u).
\label{eq:r4-block-kernel}
\end{equation}
Standalone state is $S^{\mathrm{sa}}_i=\varsigma^{\mathrm{sa}}_i(V)$ under $P^0$, giving $Q^{(i)}=\Law_{P^0}(S^{\mathrm{sa}}_i,L_i)$ with support $\mathcal S^{\mathrm{sa}}_i$.
By Lemma~\ref{lem:r4-product}(a), this equals $\Law_{P^{\mathrm{sa}}_i}(S^{\mathrm{sa}}_i,L_i)$ when $\varsigma^{\mathrm{sa}}_i$ reads only $A_i$.
Natural choices are joint-baseline declaration $\varsigma^{\mathrm{sa}}_i=\varsigma$ and interface declaration $S^{\mathrm{sa}}_i=\psi_i^{-1}(S^{(i)})$ when $\psi_i$ is bijective.

\paragraph{Conditions.}
Let $\mathcal W_i=\{j\in[N]:j<\min\mathsf C_i\}$ be all coordinates preceding the first coordinate of $\mathsf C_i$.
\par\textup{(Blk$_i$)}~\emph{Block form.} Every coordinate of $\Sigma_i$ precedes every coordinate of $\mathsf C_i$.
\par\textup{(Rd$_i$)}~\emph{The state holds the interface.} For map $\psi_i:\mathcal S\to\mathcal V_{\Sigma_i}$, $V_{\Sigma_i}=\psi_i(S)$ holds $P^\kappa$-almost surely.
\par\textup{(T$_i$)}~\emph{State fixed first.} $S$ is $P^\kappa$-almost surely a function of $V_{\mathcal W_i}$. This holds when $\Sigma^S\subseteq\mathcal W_i$, or under (Blk$_i$) and (Rd$_i$) when $\psi_i$ is injective on $\mathcal S_+$.
\par\textup{(Supp$^{\Sigma}_i$)}~\emph{Interface support.} Every $V_{\Sigma_i}$ of positive $P^\kappa$-probability lies in $\mathcal U^{\mathrm{sa}}_i$.
\par\textup{(K1)}~\emph{The state records the interface of every component.} (Rd$_i$) holds for every $i$, so $V_\Sigma$ is a function of $S$.
\par\textup{(K2)}~\emph{One component per state kernel.} Under $\kappa_{\mathrm{conn}}$, each kernel in $\Sigma$ reads coordinates of at most one block $\mathsf C_k$.

\begin{lemma}[Product Baseline]
\label{lem:r4-product}
\textup{(a)} $P^0(v)=\prod_{i=1}^nP^{\mathrm{sa}}_i(v_{A_i})$ for every $v$.
\end{lemma}
\begin{proof}[Proof sketch]
Group the product in \eqref{eq:r4-joint-base} across independent coordinate blocks $A_i$.
\end{proof}

\begin{lemma}[Component Block]
\label{lem:r4-block}
Assume \textup{(Blk$_i$)}, and let $P$ be $P^0$ or $P^\kappa$. \textup{(a)} $k^{\mathrm{blk}}_i(\cdot\mid u)$ is a probability law for every $u$. \textup{(c)} $P(V_{\mathcal W_i}=w,\ V_{Y_i}=y)=P(V_{\mathcal W_i}=w)\,k^{\mathrm{blk}}_i(y\mid w_{\Sigma_i})$ for all $w$ and $y$.
\par The lemma also holds with $Y_i$ replaced by any set between $Y_i$ and $\mathsf C_i$. For $\mathsf C_i$ itself, $k^{\mathrm{blk}}_i(c\mid u)=\prod_{j\in\mathsf C_i}k_j(c_j\mid(u,c)_{S_j})$.
\end{lemma}
\begin{proof}[Proof sketch]
(a) Summing kernels in reverse topological order telescopes to one because each factor is a conditional probability vector.
(c) By (Blk$_i$), $\mathcal W_i$ is an initial segment containing $\Sigma_i$. Coordinates outside $\mathcal W_i\cup\mathsf C_i$ sum to one in reverse order as no downstream factor reads them. The remaining product factors into $P(V_{\mathcal W_i}=w)$ and the block kernels; marginalizing over $c$ with $c_{Y_i}=y$ yields $k^{\mathrm{blk}}_i(y\mid w_{\Sigma_i})$.
\end{proof}

\begin{proposition}[Composition as a Replacement]
\label{prop:r4-composition}
In the joint specification above, assume \textup{(A1)} and \textup{(A2)} in the form \eqref{eq:r4-conn}, and consider unconditional queries.
\begin{enumerate}
\item[(i)] \emph{Invariance.} Under \textup{(Blk$_i$)}, \textup{(T$_i$)}, and \textup{(Rd$_i$)}, $F_i(\cdot\mid s)=G_i(\cdot\mid\psi_i(s))$ for every $s\in\mathcal S_+$.
\item[(ii)] \emph{Conditional independence.} Under \textup{(K1)} and \textup{(K2)}, $L_1,\dots,L_n$ are mutually conditionally independent given $S$ under $P^\kappa$, which is \textup{(CI)}. Suppose also \textup{(Blk$_i$)} and \textup{(T$_i$)} for every $i$, and let $s\in\mathcal S_+$ have $\psi_i(s)\in\mathcal U^{\mathrm{sa}}_i$ for every $i$. Then the conditional law of $(L_1,\dots,L_n)$ given $S=s$ is $\prod_iP^{\mathrm{sa}}_i(L_i\in\cdot\mid S^{(i)}=\psi_i(s))$, and Proposition~\ref{prop:component-joint}(i) gives the joint loss law on such states. Neither \textup{(K1)} nor \textup{(K2)} can be dropped.
\end{enumerate}
\end{proposition}

\begin{remark}[Standalone law, invariance, and timing]\label{rem:r4-standalone-inv}
In the joint specification above, assume \textup{(A1)} and \textup{(A2)} in the form \eqref{eq:r4-conn}, and consider unconditional queries.
(a) Under \textup{(Blk$_i$)}, $P^{\mathrm{sa}}_i(L_i\in\cdot\mid S^{(i)}=u)=G_i(\cdot\mid u)$ for every $u\in\mathcal U^{\mathrm{sa}}_i$.
(b) Suppose Proposition~\ref{prop:r4-composition}(i) applies, and the declared standalone state satisfies the counterparts of \textup{(T$_i$)} and \textup{(Rd$_i$)} under $P^0$: $S^{\mathrm{sa}}_i$ is $P^0$-almost surely a function of $V_{\mathcal W_i}$, and $\psi_i(S^{\mathrm{sa}}_i)=S^{(i)}$ holds $P^0$-almost surely. Then \textup{(Inv$_i$)} of Definition~\ref{def:p3-assumptions} holds with the kernel $G_i(\cdot\mid\psi_i(\cdot))$. The interface declaration satisfies both counterparts. The joint-baseline declaration satisfies them when $\Sigma^S\subseteq\mathcal W_i$ and $\psi_i(\varsigma(v_{\Sigma^S}))=v_{\Sigma_i}$ for every $v$, as when $S$ records $V_{\Sigma_i}$ as a coordinate.
(c) Condition \textup{(T$_i$)} cannot be dropped. A state written after component $i$ acts can break the identity in Proposition~\ref{prop:r4-composition}(i) while \textup{(Blk$_i$)}, \textup{(Rd$_i$)}, and \textup{(Supp$^{\Sigma}_i$)} hold; under the joint-baseline declaration, \textup{(Inv$_i$)} then fails while \textup{(Supp$_i$)} holds.
(d) Assume \textup{(Blk$_i$)} but not \textup{(T$_i$)}. Suppose that, for one map $g$, $S=g(V_{\Sigma_i},V_{\mathsf C_i})$ holds $P^\kappa$-almost surely and $S^{\mathrm{sa}}_i=g(V_{\Sigma_i},V_{\mathsf C_i})$ holds $P^0$-almost surely. If $V_{\Sigma_i}$ has the same law under $P^0$ and $P^\kappa$, then $(S,L_i)$ has law $Q^{(i)}$ under $P^\kappa$, so \textup{(Inv$_i$)} and \textup{(Supp$_i$)} hold.
\par\emph{Arguments.} (a) The order condition (Blk$_i$) also holds under $P^0$, so Lemma~\ref{lem:r4-block}(c) for $P^0$, summed over $w$ with $w_{\Sigma_i}=u$, and Lemma~\ref{lem:r4-product}(a) give the claim. (b) The proof of Proposition~\ref{prop:r4-composition}(i), run under $P^0$ with $S^{\mathrm{sa}}_i$ in place of $S$ and the two counterparts in place of (T$_i$) and (Rd$_i$), gives $Q^{(i)}(L_i\in\cdot\mid S=s)=G_i(\cdot\mid\psi_i(s))$ on $\mathcal S^{\mathrm{sa}}_i$. The interface declaration meets the counterparts because $S^{(i)}$ lies in $\mathcal W_i$ by (Blk$_i$) and $\psi_i(\psi_i^{-1}(S^{(i)}))=S^{(i)}$; the joint-baseline case is immediate. (c) Let component~2 have the constant state $q_2=0$, a private fair bit $X$, and $L_2=X$, and let $S=(q_1,q_2)$, where $q_1$ is generated after $X$: an independent fair bit in the standalone runs, and $q_1=X\wedge B$ for a fair bit $B$ drawn by the state kernel under $\kappa_{\mathrm{conn}}$. (Blk$_2$) and (Rd$_2$) hold with $\psi_2(s)=s_2$, which is not injective on $\mathcal S_+=\{(0,0),(1,0)\}$, and (Supp$^{\Sigma}_2$) holds. The composed law gives $P(L_2=1\mid S=(0,0))=1/3$ and $P(L_2=1\mid S=(1,0))=1$, while $G_2(1\mid0)=1/2$. Under the joint-baseline declaration both deployments have state support $\{(0,0),(1,0)\}$, so (Supp$_2$) holds, and the standalone runs give $1/2$ at both states, so (Inv$_2$) fails. (d) Lemma~\ref{lem:r4-block}(c) with $\mathsf C_i$ in place of $Y_i$ gives $P(V_{\Sigma_i}=u,V_{\mathsf C_i}=c)=P(V_{\Sigma_i}=u)\prod_{j\in\mathsf C_i}k_j(c_j\mid(u,c)_{S_j})$ under either law; the product is unchanged by (A1) and the law of $V_{\Sigma_i}$ by hypothesis (by Lemma~\ref{lem:p2-truncation} when no coordinate of $\Sigma_i$ is downstream). Since $(S,L_i)$ and $(S^{\mathrm{sa}}_i,L_i)$ are the same function of $(V_{\Sigma_i},V_{\mathsf C_i})$, $\Law_{P^\kappa}(S,L_i)=Q^{(i)}$; its state marginal gives $\mathcal S_+=\mathcal S^{\mathrm{sa}}_i$ and its conditional laws give (Inv$_i$).
\end{remark}

\begin{proof}[Proof of Proposition~\ref{prop:r4-composition}]
(i) By (T$_i$), $S=\tau_i(V_{\mathcal W_i})$ almost surely, with $\Sigma_i\subseteq\mathcal W_i$ by (Blk$_i$). Summing Lemma~\ref{lem:r4-block}(c) over $w$ with $\tau_i(w)=s$ and applying (Rd$_i$) yields $P^\kappa(S=s,V_{Y_i}=y)=P_S(s)k^{\mathrm{blk}}_i(y\mid\psi_i(s))$. On $\{S=s\}$, $L_i=\ell_i(\psi_i(s),V_{Y_i})$, so $F_i(\cdot\mid s)=G_i(\cdot\mid\psi_i(s))$, where $k^{\mathrm{blk}}_i$ is unchanged by (A1).
(ii) Given $S=s$, (K1) fixes $V_\Sigma=\psi(s)$. By (K2), each state kernel in $\Sigma$ reads at most one block $\mathsf C_k$, and block kernels read only $\Sigma_k\cup\mathsf C_k$. Thus the joint law factors as $\prod_k\phi_k(c_{\mathsf C_k})$, establishing mutual conditional independence given $S$. Under (Blk$_i$) and (T$_i$), part (i) and Remark~\ref{rem:r4-standalone-inv}(a) identify each factor where $\psi_i(s)\in\mathcal U^{\mathrm{sa}}_i$.
Neither condition can be dropped. Without (K2), let components~1 and~2 have constant states and independent fair bits, and let component~3 read shared spend $S=L_1+L_2$: (K1) holds, yet $P(L_1=L_2=1\mid S=1)=0$ versus product $1/4$. Without (K1), let fair tape $T$ be a state coordinate and set $q_2=L_1\oplus T$, $q_3=L_2\oplus T$: given $S=(q_1,q_2,q_3)$, $L_1\oplus L_2=q_2\oplus q_3$, inducing dependence.
\end{proof}
```
