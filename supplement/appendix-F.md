# Supplementary Material for Appendix F.1 to F.7: Supported Continuation, Unsupported Inputs, and Model Updates

This companion document holds coordinate-model details, supporting proofs, and remarks for Appendices F.1 to F.7 of "agent-risk: Risk Assessment for Changes to AI Agents".

---

## F.1 Detailed Coordinate Model and Tree Embedding

### Original Setting and Paragraphs

#### Paths and Kernels
Coordinates of a joint execution and observation path in generation order are $V_1,\dots,V_n$ with finite ranges $\mathcal V_1,\dots,\mathcal V_n$, and $\Omega=\prod_i\mathcal V_i$. For $S\subseteq[n]=\{1,\dots,n\}$, $V_S=(V_i)_{i\in S}$, $\mathcal V_S=\prod_{i\in S}\mathcal V_i$, and $v_S$ is the restriction of $v\in\Omega$ to $S$. Kernel $i$ has declared input set $S_i\subseteq[i-1]$ and conditional law $k_i(\cdot\mid v_{S_i})$ on $\mathcal V_i$. Coordinates include execution, observation, and random-input kernels. The baseline law is
$$
P^0(v)=\prod_{i=1}^{n}k_i(v_i\mid v_{S_i}),\qquad v\in\Omega .
$$

#### Declared Kernel Values
A declared set $\mathcal K_i\subseteq\mathcal V_{S_i}$ collects inputs where $k_i$ is known by declaration (default $\mathcal K_i=\emptyset$); inactive slots emit a dummy value, and post-abort valuation slots carry the valuation of (A3) in Proposition A.4.1.

#### Embedding the Tree Setting
Tree mechanisms of Proposition A.4.1 embed into coordinates by allocating one slot per history where call order or kernel identity varies. Inactive slots emit a dummy value. Replaced slots are those where replaced and baseline kernels differ. The post-abort path is a single valuation slot carrying the law of $L$ given the prefix (declared as $b_\kappa$ after terminal deviations by (A3)). Allowed sets $\mathcal E_{j,z}$ exclude the dummy at active inputs, and equal $\{\ell(\omega):\omega\text{ extends }p\}$ at undeclared abort prefixes $p$.

#### Replacement
Replacement $\kappa$ acts at boundary $s$, replacing kernels $D_\kappa\subseteq\{s,\dots,n\}$ by known kernels $k^\kappa_j(\cdot\mid v_{S^\kappa_j})$ with $S^\kappa_j\subseteq[j-1]$ (with $k^\kappa_j=\delta_{f^\kappa_j(v_{S^\kappa_j})}$ for deterministic replacements). For $i\notin D_\kappa$, $k^\kappa_i=k_i$ and $S^\kappa_i=S_i$. Invariance and prefix preservation ((A1) and (A2)) give
$$
P^\kappa(v)=\prod_{i=1}^{n}k^\kappa_i(v_i\mid v_{S^\kappa_i}).
$$
Here $P^0=P_{E,K}$ and $P^\kappa=P_{E^\kappa,K^\kappa}$ on $\Omega$.

#### Loss and Records
Loss is $L=\ell(V)=\bar\ell(V_{S_L})$ for $S_L\subseteq[n]$ and $\bar\ell\colon\mathcal V_{S_L}\to\mathbb L$. Records are $V_{\mathcal O}$ for $\mathcal O\subseteq[n]$. The pre-intervention record is $R_t=V_{S_r}$ with $S_r\subseteq\mathcal O\cap[s-1]$ and $P^0(R_t=r)>0$ ($t=-1, S_r=\emptyset$ for unconditional queries). A value $v$ is $r$-compatible when $v_{S_r}=r$. The continuation query is $Q^\kappa_t(B\mid r)=P^\kappa(L\in B\mid R_t=r)$ for $B\subseteq\mathbb L$.

---

## F.2 Deployment Interpretation of Support Assumptions

### Original Text
For an agent whose input is its full text context, (G1) asks that every context the change leads an unchanged kernel to read has positive baseline probability at that position. A refusal message or regenerated text that baseline operation never produces fails it. Declaring that the kernel reads only a coarsening of its context creates support only under that sufficient-input assumption, which logs alone cannot confirm.

---

## F.3 Extended Proofs of Structural Lemmas

### Original Proof of Lemma F.3.1 (Truncated Factorization)
Summing out coordinates in reverse index order, downstream factors sum to one because each factor depends only on earlier coordinates. Unaffected factors depend only on $\bar\Delta_\kappa$ and coincide under both laws by (A1) and (A2), yielding agreement on $V_{\bar\Delta_\kappa}$ and $R_t$. For (11), summing (10) over $\Delta_\kappa\setminus K$ leaves only factors in $K$ since $K$ is $\kappa$-closed, recovering standard truncated factorization (Robins, 1986; Pearl and Robins, 1995; Pearl, 2009).

---

## F.4 Detailed Proof of the Supported-Continuation Proposition

### Original Proof Sketch of Proposition A.5.1 (Section F.4)
Parts (i) and (ii) apply the sequential g-formula (Robins, 1986; Pearl and Robins, 1995) in coordinate order. By induction on $m\leq q$, Lemmas F.3.1 and F.3.2 give $P^\kappa(V_{J_\kappa}=v)=G(v)$ for $r$-compatible $v$, yielding (13); the full induction appears in Proposition F.5.2(a),(c). Because $J_\kappa\subseteq\mathcal O$ under (G2) to (G4), each factor in $G_m$ is determined by the baseline record law, detecting any failure of (G1) and proving (ii). For (iii), record all coordinates of $U\sim\mathrm{Bernoulli}(1/2)$, baseline $A=U$, replacement $A=1-U$, and $Y$ whose kernel reads only $A$, with $P(Y=1\mid A=0)=1/4$ and $P(Y=1\mid A=1)=3/4$; take $L=Y$ and no conditioning record. Both outcome inputs have baseline support, all decisions disagree, and the g-formula gives replacement risk $1/2$. For (iv), agreement paths preserve the baseline law by (A1) and (A2), while terminal abort paths have declared inputs by (A3); subtracting the agreement and terminal restrictions of Proposition A.4.1 identifies the remainder. Part (v) is Proposition F.5.2.

---

## F.5 Full Proofs and Extended Remarks on Unsupported Inputs

### Original Setup for Free Arguments at Null Inputs
In the coordinate model of Appendix F.1, let $\mathcal U_\kappa=J_\kappa\cap(\Delta_\kappa\setminus D_\kappa)$. For $j\in\mathcal U_\kappa$ and $z\in\mathcal V_{S_j}\setminus\mathcal K_j$, kernel $j$ emits values in a nonempty allowed set $\mathcal E_{j,z}\subseteq\mathcal V_j$ (default $\mathcal V_j$). Members of $\mathcal M_0$ share the declared inputs, declared kernel values, known replacement kernels, and allowed sets. Each coordinate has its own kernel; restrictions tying slots are addressed in Remark F.5.4. The population identified set is
$$
\Theta_r=\bigl\{Q^\kappa_{t,M}(\cdot\mid r):\ M\in\mathcal M_0,\ V_{\mathcal O}\text{ has the observed record law under }P^0_M\bigr\}.
$$
Under (G2), an input $z\in\mathcal V_{S_j}\setminus\mathcal K_j$ with $P^0(V_{S_j}=z)=0$ is a null input; collect them in $\mathcal Z_j$. The null blocks are $\mathcal B=\{(j,z):j\in\mathcal U_\kappa,\ z\in\mathcal Z_j\}$, and free families form $\Lambda=\prod_{(j,z)\in\mathcal B}\Delta(\mathcal E_{j,z})$. For a member $M$ with the observed record law, define its own null-input family by $\lambda_M=(k^M_j(\cdot\mid z))_{(j,z)\in\mathcal B}$. For $\lambda\in\Lambda$, the recursion of Appendix F.4 defines $G^\lambda$ via (12), and
$$
Q^\lambda(B\mid r)=\frac{1}{P^0(R_t=r)}\sum_{v\in\mathcal V_{J_\kappa}:\ v_{S_r}=r,\ \bar\ell(v_{S_L})\in B}G^\lambda(v).
$$
The class $\mathcal M_0$ is *closed at null inputs* if for each member $M$ with the observed record law and each $\lambda\in\Lambda$, setting $k_j(\cdot\mid z)=\lambda_{j,z}$ on $\mathcal B$ yields a member $M^\lambda\in\mathcal M_0$.

### Original Proof of Proposition F.5.2(a)
Fix such an $M$, and compute with its record law, its null blocks, and $\lambda=\lambda_M$. As noted before Lemma F.3.1, $J_\kappa\subseteq\mathcal O$. By the closure property and $S^\kappa_{j_l}\subseteq[j_l-1]$, every element of $S^\kappa_{j_l}$ lies in $J_\kappa\cap\bar\Delta_\kappa$ or in $\{j_1,\dots,j_{l-1}\}$, so $S^\kappa_{j_l}\subseteq K_{l-1}$. Hence each $K_m$ is $\kappa$-closed, and Lemma F.3.1 gives, for every $v$,
$$
\Pi_m(v):=P^\kappa_M\!\left(V_{K_m}=v_{K_m}\right)=P^0_M\!\left(V_{K_0}=v_{K_0}\right)\prod_{l\leq m}k^\kappa_{j_l}\!\left(v_{j_l}\mid v_{S^\kappa_{j_l}}\right). \tag{16}
$$
We show $G^{\lambda_M}_m(v)=\Pi_m(v)$ for every $r$-compatible $v$ by induction on $m$; the argument uses neither (G1) nor Proposition A.5.1. The case $m=0$ holds by definition. Suppose the claim holds for $m$, and let $j=j_{m+1}$. If $\Pi_m(v)=0$, both sides vanish at $m+1$. Otherwise the factor is computed by one of the four cases of (12). If $j\in D_\kappa$, it is $k^\kappa_j(v_j\mid v_{S^\kappa_j})$ by definition. At a declared input $v_{S_j}\in\mathcal K_j$, it is the declared value $k_j(v_j\mid v_{S_j})$, which $M$ shares. In the second case, $P^0_M(V_{S_j}=v_{S_j})>0$ by the definition of $\mathcal Z_j$, so Lemma F.3.2 with $K=S_j$ gives $c^{\lambda_M}_j(v)=k^M_j(v_j\mid v_{S_j})$. The fourth case gives $c^{\lambda_M}_j(v)=k^M_j(v_j\mid v_{S_j})$ by the definition of $\lambda_M$. Since $j\notin D_\kappa$ in these last three cases, (A1) and (A2) give $k^M_j(v_j\mid v_{S_j})=k^\kappa_j(v_j\mid v_{S^\kappa_j})$. Each case therefore gives $G^{\lambda_M}_{m+1}(v)=\Pi_m(v)\,k^\kappa_j(v_j\mid v_{S^\kappa_j})=\Pi_{m+1}(v)$ by (16). At $m=q$, $G^{\lambda_M}(v)=P^\kappa_M(V_{J_\kappa}=v)$. Since $S_L\cup S_r\subseteq J_\kappa$, summing over $r$-compatible $v$ with $\bar\ell(v_{S_L})\in B$ and dividing by $P^\kappa_M(R_t=r)=P^0_M(R_t=r)$ (Lemma F.3.1) gives (15) with $\lambda=\lambda_M$. Every term other than $\lambda$ is a record-law functional or a declared or known kernel, and the sets $\mathcal Z_j$ are record-law functionals by (G2). A member with the observed record law has the observed null blocks and $\lambda_M\in\Lambda$, which gives the inclusion.

### Original Proof of Proposition F.5.2(b)
Let $\lambda\in\Lambda$, and let $M$ be a member with the observed record law. By closure, $M^\lambda$ is a member. For each $j$ with a null block, $M^\lambda$ changes kernel $j$ only on $\mathcal Z_j$, which has baseline probability zero. Apply Lemma F.5.1 once for each such $j$; after each application the baseline law is unchanged, so the next set $\mathcal Z_{j'}$ still has probability zero. Hence $P^0_{M^\lambda}=P^0_M$, so $M^\lambda$ has the observed record law, the same null blocks, and $\lambda_{M^\lambda}=\lambda$. Part (a) gives $Q^\kappa_{t,M^\lambda}(\cdot\mid r)=Q^\lambda(\cdot\mid r)$, so the image lies in $\Theta_r$. Part (a) also gives the reverse inclusion.

### Original Proof of Proposition F.5.2(c)
Let $M$ be a member with the observed record law that satisfies (G1), let $\Pi_m$ be as in the proof of (a), and let $G_m$ be built with every $\mathcal Z_j$ empty, as in Appendix F.4. For every $\lambda\in\Lambda$ we show $G^\lambda_m=G_m=\Pi_m$ on $r$-compatible values by induction on $m$; the case $m=0$ holds by definition. If $\Pi_m(v)=0$, all three vanish at $m+1$. Otherwise let $j=j_{m+1}$. If $j\in D_\kappa$ or $v_{S_j}\in\mathcal K_j$, the factors coincide with those of the proof of (a). If $j\notin D_\kappa$ and $v_{S_j}\notin\mathcal K_j$, then $S_j\cup S_r\subseteq K_m$, so $P^\kappa_M(V_{S_j}=v_{S_j},\ R_t=r)\geq\Pi_m(v)>0$, and (G1) gives $P^0(V_{S_j}=v_{S_j})>0$. Hence $v_{S_j}\notin\mathcal Z_j$, all three formulas use the second case of (12), which is defined, and the proof of (a) gives the step. At $m=q$, (15) equals the right side of (13) for every $\lambda\in\Lambda$.

### Original Proof of Proposition F.5.2(d) (Archived; Not Printed in Paper)
For $r$-compatible $v$, every factor in (12) is defined, and a factor depends on $\lambda$ only in the fourth case, through the coordinate $\lambda_{j,v_{S_j}}(v_j)$ of the block $(j,v_{S_j})$; different factors have different indices $j$ and hence different blocks. Summing over $v$, the map $\lambda\mapsto Q^\lambda(B\mid r)$ is a polynomial that is affine in each block when the other blocks are fixed, and so is the mean $\sum_xx\,Q^\lambda(\phi^{-1}(x)\mid r)$. For such a polynomial $f$ and $\lambda\in\Lambda$, replacing the blocks one at a time by a vertex $\delta_a$ of $\Delta(\mathcal E_{j,z})$ that does not decrease $f$ gives a point-mass family $\lambda'$ with $f(\lambda')\geq f(\lambda)$, and likewise one with a value at most $f(\lambda)$. The set $\Lambda$ is a finite product of simplices, hence compact and convex, and $\lambda\mapsto Q^\lambda(\cdot\mid r)$ is continuous, so the image is compact and connected and the range of $f$ is the stated closed interval.

### Original Text of Former Remark F.5.3 (Two Extremes and What Lies Between; Archived, Not Printed in Paper)
Part (c) gives a single law under (G1). At the other end, with $\mathcal O=[n]$ and the embedding with its tree-feasible allowed sets, the image always lies in the set $\mathcal S_\phi(P^0)$ of Proposition E.4.3. Take $\lambda\in\Lambda$ and a member $M$ with the observed record law, here the full baseline law $P^0$. At a replaced slot $d$, Lemma F.3.2 with $K=S_d$ makes $k^M_d$ the kernel of the embedded tree mechanism, $\delta_{f^0}$ or a declared dummy, at inputs of positive baseline probability. Setting $k^M_d$ to that kernel at the other inputs leaves $P^0_M$ unchanged by Lemma F.5.1, and $P^\kappa_M$ uses $k^\kappa_d$ instead, so we may assume that $M$ satisfies (R). It satisfies (A1) and (A2) through (10), and (A3) because members share the declared valuation. The mechanism $M^\lambda$ then emits only values that tree paths produce and satisfies (R) and (A1) to (A3), whether or not it belongs to $\mathcal M_0$, with baseline law $P^0$ by Lemma F.5.1. By part (a), $Q^\lambda$ is its continuation law, and Proposition E.4.3 places $\phi_\#Q^\lambda$ in $\mathcal S_\phi(P^0)$. Proposition E.1.1 thus remains a valid outer bound, and the image of Proposition F.5.2 is never wider.

### Original Text of Remark F.5.3 (What Finite Logs Show)
Under (G2) to (G4), the record law decides which inputs are null (Proposition A.5.1(ii)). A finite log shows support for each input it contains, but absence from a log does not establish a null input. Treating such an input as null still gives a valid outer set: setting the added argument to the baseline conditional law at that input returns the original factors of (12), so the enlarged image contains the original one. Finite-sample inference is a separate claim (Limitations).

### Original Text of Remark F.5.4 (Scope of Sharpness)
Part (b) uses that each coordinate's kernel is free at its own null inputs, and part (d) describes the identified set through part (b). In the slot embedding of Appendix F.1, several slots can host one named kernel. A class that ties slots, such as one law per kernel and step when the embedding needs more slots than kernel and step pairs, is covered by the outer-set clause of Proposition A.5.1. It is a subclass with the same evidence, so part (a) still gives an outer set for it. Its identified set can be smaller, and its ranges need not be attained at point masses. Let one kernel be called twice on a path, with outputs $N_1,N_2\in\{0,1\}$, one law $\mathrm{Bern}(y)$ at a null input that both calls reach, and $L=N_1+N_2$. Then $P(L=1)=2y(1-y)$ attains its maximum $1/2$ only at $y=1/2$, and both point masses give $0$.

---

## F.6 Detailed Proof for Model Updates

### Original Proof Sketch of Corollary F.6.1
On the agreement set $\mathsf A_\kappa$, the baseline and replacement policies produce identical outputs at each decision step, so (A1) and (A2) give $P^\kappa(v)=P^0(v)$ for all $v\in\mathsf A_\kappa$, identifying $\mathsf m_\kappa(r)=P^0(\mathsf A_\kappa\mid R_t=r)$ and the agreement loss contribution when loss values are logged.
Because decision inputs and random tapes are recorded, seed-addressable evaluation determines the agreement indicator of each logged episode directly from its record.
The empirical agreement frequency over independent baseline episodes is therefore an unbiased binomial estimator that converges almost surely to $\mathsf m_\kappa(r)$ by the strong law of large numbers.
Finally, agreement through all decisions implies agreement at the first decision, yielding the bound $\mathsf m_\kappa(r)\leq P^0(f^\kappa_1(I_1)=A_1\mid R_t=r)$, which requires evaluating the new policy at the first logged decision only.

---

## F.7 Extended Proofs and Details for Budgeted Targeted Execution and Restoration Allowances

### Original Setting and Bounds Prose
Targeted execution need not run every continuation. Under replay, the agree and stop terms of display (2) come from the log, and only continuing episodes whose outcome the log leaves open cost execution. A uniform sample of them bounds the rest. The inference below is classical (hypergeometric and binomial inversion (Clopper and Pearson, 1934), a union bound, and error spending over declared looks (Lan and DeMets, 1983)); its use here is to decide how many continuations a decision needs.

Let harm be binary and $q$ the changed system's P(harm), the target $q^\kappa_t(r)$ with $X$ the harm indicator. A pass-all log holds $n$ episodes. Under (R), (A1), (A2), and (A3), replay resolves an episode's outcome under the change when it agrees, when harm occurred before its first deviation, or when it stops and the abort valuation fixes the harm indicator (a point mass, as when a stop ends the episode with the harm accrued before it); $b$ resolved episodes are harmful. A nondegenerate abort valuation would need an independent draw for each stopped episode. The remaining $N$ continuing episodes are unresolved, partitioned into strata $s$ of sizes $N_s$ fixed by the log (for example, by the step of first deviation). Executing an unresolved episode's continuation from its snapshot gives its outcome under the change, by (T). The completed cohort assigns each resolved episode its replayed outcome and each unresolved episode the outcome of its continuation; $S$ is its harm count.

Hypothesis (H): The completed outcomes are independent Bernoulli variables with common probability $q$; the random order in which unresolved episodes are executed is drawn independently of them, and the looks are fixed in advance. Independently sampled tasks and fresh seeds support (H); (T) links each continuation to its live law but does not by itself make outcomes independent across episodes, so dependence through shared serving or environment state lies outside it.

In stratum $s$, execute a uniform random sample without replacement of $k_s$ unresolved episodes and observe $x_s$ harms. With $\mathrm{H}(N_s,D,k_s)$ the hypergeometric law of harms in such a sample when $D$ of the $N_s$ are harmful, let
$$
D_s^U(x_s,k_s;e)=\max\bigl\{D\in\{x_s,\ldots,N_s-k_s+x_s\}:\ \Pr_{\mathrm{H}(N_s,D,k_s)}(X\le x_s)>e\bigr\},
$$
with $D_s^U=N_s$ when $k_s=0$; at $k_s=N_s$ it returns $x_s$. Let $S^U=b+\sum_s D_s^U$ and let $u_{\mathrm{CP}}(z,n;a)$ be the one-sided Clopper-Pearson upper limit for $z$ successes in $n$ trials at level $1-a$, with $u_{\mathrm{CP}}(n,n;a)=1$.

### Original Proof of Proposition F.7.1
(i) Fix a stratum with true harmful count $D$. The probability $\Pr_{\mathrm{H}(N_s,D,k_s)}(X\le x)$ is nonincreasing in $D$, so $D>D_s^U(x)$ implies $\Pr_{\mathrm{H}(N_s,D,k_s)}(X\le x)\le e_s$. The set of such $x$ is a lower tail of $X$ with probability at most $e_s$ under the true law, so $\Pr(D>D_s^U)\le e_s$. A union bound over strata gives $\Pr(S>S^U)\le\sum_se_s$.
(ii) Under (H), $S$ is binomial with parameters $n$ and $q$, so $\Pr(q>u_{\mathrm{CP}}(S,n;a_L))\le a_L$. Since $u_{\mathrm{CP}}$ is nondecreasing in its first argument, on the event $\{S\le S^U\}\cap\{q\le u_{\mathrm{CP}}(S,n;a_L)\}$ we have $q\le u_{\mathrm{CP}}(S^U,n;a_L)$. The bound in (i) holds given the cohort, hence unconditionally, and a union bound gives the claim; it needs no independence between the two events.
(iii) At each declared look the executed set is a uniform sample of fixed size, so (i) gives $\Pr(S>S^U_j)\le\sum_se_{s,j}$ unconditionally. A union bound over looks gives $\Pr(S>S^U_j\text{ for some }j)\le a_E$. On the complement intersected with the binomial event of (ii), $q\le u_{\mathrm{CP}}(S,n;a_L)\le u_{\mathrm{CP}}(S^U_j,n;a_L)$ for every $j$, hence $q\le\overline U_j$. The stopping rule only selects a look, and the targets only read the same sequence.

### Original Proof of Lemma F.7.2 (Restoration Error)
By the proof of Proposition E.4.3, $\phi_\#P^\kappa(L\in\cdot\mid r)=\nu+\sum_hw_hQ_h$, where $w_h=P^0([h]\mid r)$ by Lemma E.3.1(b) and $\nu$, the agree and stop terms, depends on $P^0$ and $b_\kappa$ only (Proposition A.4.1). Restoration enters through the $\hat Q_h$ alone, so the error at $B$ is $\sum_hw_h(\hat Q_h(B)-Q_h(B))$, at most $\sum_hw_h\gamma_h$ in absolute value. For the mean, $\hat Q_h-Q_h$ has total mass $0$, so $\int x\,d(\hat Q_h-Q_h)=\int_{\min\mathcal X}^{\max\mathcal X}(\hat Q_h-Q_h)(X>s)\,ds$, at most $(\max\mathcal X-\min\mathcal X)\gamma_h$ in absolute value. Weighting by $w_h$ and summing gives both bounds.

### Original Text of Remark F.7.3 (Restoration Allowance)
Proposition F.7.1 bounds the harm probability $\hat q$ of $\hat P$ when the completed restored outcomes satisfy hypothesis (H): independent Bernoulli with a common probability. Let $u$ be its certified bound and $\xi=\sum_hw_h\gamma_h$. Lemma F.7.2 gives $|q-\hat q|\le\xi$, so with the probability of the proposition $q\le u+\xi$, and a target $q^\star$ is cleared when $u+\xi\le q^\star$; exact (T) is the case $\xi=0$. For a tolerance $\xi_0\ge\xi$ declared in advance, stopping at the first look with $\overline U_j+\xi_0\le q^\star$ keeps the coverage of part (iii). The continuing mass $\varepsilon_\kappa$ is estimated from the log, so a population statement uses an upper confidence limit for it, whose error adds to that of the proposition, or the bound $\varepsilon_\kappa\le1$. At prefixes with harm before the deviation, $Q_h=\hat Q_h$ for harm, so they drop out of the harm allowance; task success can still depend on the continuation, so a decision with a success floor needs its own allowance from the same lemma. Both bounds of the lemma are attained: with $Q_h=\delta_{\min\mathcal X}$ and $\hat Q_h=(1-\gamma)\delta_{\min\mathcal X}+\gamma\delta_{\max\mathcal X}$ for every $h$, the errors equal $\varepsilon_\kappa\gamma$ and $(\max\mathcal X-\min\mathcal X)\varepsilon_\kappa\gamma$. A paired live run measures $q-\hat q$ itself and does not bound $\gamma_h$, since errors at different prefixes can offset. `checks/verify_restoration.py` checks the lemma exhaustively on a finite retry mechanism and the E1 restoration tolerances of supplement/appendix-D.md (Restoration Allowance).
