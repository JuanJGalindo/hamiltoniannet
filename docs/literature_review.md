# Literature Review — Hamiltonian Neural Networks

## 1. Core idea

The scalar-Hamiltonian trick parameterizes the *energy* $H_\theta(q,p):\mathbb{R}^{2n}\to\mathbb{R}$ with a network and *derives* the vector field as its symplectic gradient $\dot z = J^{-1}\nabla H_\theta$, $J=\begin{smallmatrix}0&I\\-I&0\end{smallmatrix}$, rather than regressing the field directly. Because $\tfrac{d}{dt}H_\theta = \nabla H_\theta^\top J^{-1}\nabla H_\theta = 0$ identically (skew-symmetry of $J$), the learned energy is conserved along every predicted trajectory by construction. A baseline MLP fitting $\dot x = f_\theta(x)$ has no such constraint: its $2n$ free field components accumulate per-step error and let trajectories drift off the true energy level set over long horizons.

## 2. Variants

### HNN — Greydanus, Dzamba & Yosinski (2019)
Establishes the paradigm: learn the scalar $H$, not the field, so a conserved energy-like quantity exists by construction.
```math
\dot q = \tfrac{\partial H_\theta}{\partial p},\quad \dot p = -\tfrac{\partial H_\theta}{\partial q} \qquad\text{(Hamilton's equations via autodiff)}
```
```math
\mathcal{L} = \big\|\tfrac{\partial H_\theta}{\partial p}-\dot q\big\|_2^2 + \big\|\tfrac{\partial H_\theta}{\partial q}+\dot p\big\|_2^2 \qquad\text{(point-wise field matching)}
```
**Data:** phase-space pairs $(q,p)$ with explicit derivatives $(\dot q,\dot p)$ (analytic or finite-difference).
**`hnet`:** `models/hnn.py — implemented`

### HGN — Toth et al. (2020)
Learns Hamiltonian dynamics from raw pixel sequences into an abstract latent phase space of arbitrary dimension.
```math
q_\phi(z\mid x_{0:T})\xrightarrow{f_\psi} s_0 \xrightarrow{\text{leapfrog }H_\gamma} s_{0:T}\xrightarrow{d_\theta}\hat x_t \qquad\text{(encode}\to\text{rollout}\to\text{decode)}
```
```math
\mathcal{L}=\tfrac{1}{T+1}\!\sum_t \mathbb{E}_{q_\phi}[\log p(x_t\mid q_t)] - \mathrm{KL}(q_\phi\|p) \qquad\text{(temporally-extended ELBO)}
```
**Data:** sequences of high-dimensional images; no coordinates, derivatives, or phase-space dimension known.
**`hnet`:** `models/hgn.py — roadmap`

### pHNN — Desai, Roberts, Mattheakis, Sondak & Protopapas (2021)
Adds energy dissipation and time-dependent forcing via the port-Hamiltonian decomposition $H{+}F(t){+}N$.
```math
\begin{smallmatrix}\dot q\\\dot p\end{smallmatrix}=\Big[J+\begin{smallmatrix}0&0\\0&N\end{smallmatrix}\Big]\nabla H + \begin{smallmatrix}0\\F(t)\end{smallmatrix} \qquad\text{(damping }N\text{, forcing }F)
```
```math
\mathcal{L}=\|\hat{\dot q}-\dot q\|^2+\|\hat{\dot p}-\dot p\|^2+\lambda_F\|F\|_1+\lambda_N\|N\|_1 \qquad\text{(L1 enforces parsimony)}
```
**Data:** derivative labels $(\dot q,\dot p)$ over input $[q,p,t]$; optional embedded-RK4 state-to-state variant.
**`hnet`:** `models/phnn.py — roadmap`

### Adaptable HNN — Han, Glaz, Haile & Lai (2021)
Adds a parameter input channel so one HNN predicts dynamics at *unseen* bifurcation parameters (routes to chaos).
```math
H_\theta(q,p,\alpha),\quad \dot q=\tfrac{\partial H_\theta}{\partial p},\ \dot p=-\tfrac{\partial H_\theta}{\partial q} \qquad\text{($\alpha$ conditions, not differentiated)}
```
**Data:** derivative data $(\dot q,\dot p)$ at $\gtrsim 4$ distinct parameter values; interpolates/extrapolates over $\alpha$.
**`hnet`:** `models/adaptable_hnn.py — roadmap`

### Self-Supervised HNN — Mattheakis et al. (2022)
Inverts the problem: with $H$ *known*, the network is a data-free solver finding trajectories satisfying Hamilton's equations.
```math
\hat z(t)=z(0)+(1-e^{-t})\,N(t;\theta) \qquad\text{(trial form enforces IC exactly)}
```
```math
\mathcal{L}=\tfrac1K\!\sum_n\|\dot{\hat z}^{(n)}-J\nabla H(\hat z^{(n)})\|^2+\lambda\,(H(\hat z^{(n)})-E_0)^2 \qquad\text{(PDE residual + energy reg.)}
```
**Data:** none — only analytic $H$, initial state $z(0)$, and $K$ collocation times in $[0,T]$.
**`hnet`:** `models/selfsup_hnn.py — roadmap`

### SHNN — David & Méhats (2023)
Fixes HNN's implicit forward-Euler loss (non-symplectic → irremovable loss floor) by embedding a *symplectic* scheme.
```math
\mathcal{L}=\big\|\tfrac{y_1-y_0}{h}-J^{-1}\nabla\tilde H(s(y_0,y_1))\big\|^2 \qquad\text{($s$: sympl. Euler or midpoint)}
```
```math
H=\tilde H-\tfrac{h}{2}\nabla_p\tilde H\!\cdot\!\nabla_q\tilde H+\mathcal{O}(h^2) \qquad\text{(backward-error recovery of true }H)
```
**Data:** discrete state pairs $(y_0,y_1{=}\varphi_h(y_0))$ at fixed step $h$; *no* derivative labels needed.
**`hnet`:** `models/shnn.py — roadmap`

### sPHNN — Roth, Klein, Kannapinn, Peters & Weeger (2025)
Adds provable global Lyapunov stability to port-Hamiltonian learning via a convex (FICNN) Hamiltonian.
```math
\dot x=[J(x)-R(x)]\tfrac{\partial H}{\partial x}+G(x)u(t) \qquad\text{(}R=LL^\top\succeq0\text{ dissipation)}
```
```math
H(x)=f(x)-f(x^*)-\nabla f|_{x^*}^\top(x-x^*)+\varepsilon\|x-x^*\|^2 \qquad\text{(convex }f\Rightarrow\text{Lyapunov)}
```
**Data:** derivative pairs or trajectory data with input $u(t)$ (e.g. real cascaded-tanks, 1024 pts).
**`hnet`:** `models/sphnn.py — roadmap`

### SNO — Makara, Tanaka, Matsubara & Yaguchi (2026)
Extends symplectic learning to infinite dimensions (Hamiltonian PDEs); the operator's flow map is symplectic by construction.
```math
\Phi_V(q,p)=(q,\,p+\nabla V(q)),\quad \Psi_W(q,p)=(q+\nabla W(p),\,p) \qquad\text{(symplectic shear maps)}
```
```math
\theta^\star=\arg\min_\theta\tfrac1N\!\sum_i\|N_\theta(u^{(i)})-v^{(i)}\|_{\mathcal P}^2,\ v\approx\Phi^\tau(u) \qquad\text{(SAFNO-parameterized }V,W)
```
**Data:** consecutive state-*function* pairs $(u_n,u_{n+1})$ at step $\tau$; mesh-independent, $H$-functional unknown.
**`hnet`:** `models/sno.py — roadmap`

## 3. Comparison table

| Variant             | Year | System class                | Symplectic                 | Needs ẋ labels       | Solver in train loop      | hnet module               |
| ------------------- | ---- | --------------------------- | -------------------------- | -------------------- | ------------------------- | ------------------------- |
| HNN                 | 2019 | Conservative, autonomous    | Yes (by construction)      | Yes                  | No                        | `models/hnn.py`           |
| HGN                 | 2020 | Conservative, from pixels   | Approx (leapfrog latent)   | No                   | Yes (leapfrog)            | `models/hgn.py`           |
| pHNN                | 2021 | Forced + damped             | No (dissipation modeled)   | Yes (or state pairs) | Optional (RK4)            | `models/phnn.py`          |
| Adaptable HNN       | 2021 | Conservative, param-varying | Yes                        | Yes                  | No                        | `models/adaptable_hnn.py` |
| Self-Supervised HNN | 2022 | Conservative, known $H$     | Yes (NN is solver)         | No (data-free)       | No (NN is solver)         | `models/selfsup_hnn.py`   |
| SHNN                | 2023 | Conservative, autonomous    | Yes (strict)               | No (state pairs)     | Yes (symplectic, in loss) | `models/shnn.py`          |
| sPHNN               | 2025 | Forced + damped + stable    | No (dissipation by design) | Optional             | Yes (Tsit5, traj. fit)    | `models/sphnn.py`         |
| SNO                 | 2026 | Conservative PDEs (∞-dim)   | Yes (strict)               | No (function pairs)  | No (SNO is stepper)       | `models/sno.py`           |

## 4. Lineage

- **HNN → HGN:** lifts the phase-space-data requirement — learns an arbitrary-dimensional latent Hamiltonian directly from pixels via a VAE + leapfrog rollout.
- **HNN → pHNN:** removes the conservative-only restriction — port-Hamiltonian split adds learnable dissipation $N$ and forcing $F(t)$, with L1 collapsing to plain HNN when truly conservative.
- **HNN → Adaptable HNN:** removes single-parameter training — a separate $\alpha$ channel makes one model generalize across bifurcation parameters while preserving the symplectic structure.
- **HNN → Self-Supervised HNN:** inverts the data direction — given known $H$, learns trajectories with zero data, conserving the *original* (not a perturbed) energy.
- **HNN → SHNN:** removes the forward-Euler loss floor — a symplectic scheme in the loss lets the network learn an exact modified Hamiltonian, recoverable to true $H$ by backward-error analysis.
- **pHNN → sPHNN:** adds stability guarantees and a full state-dependent dissipation matrix $R(x)$ — a convex FICNN Hamiltonian makes $H$ a Lyapunov function (bounded trajectories, unique attractor).

## 5. Paper corpus

- **HNN (Greydanus 2019)** — *Hamiltonian Neural Networks* — root model; learn scalar $H$, conserve energy by construction.
- **HGN (Toth 2020)** — *Hamiltonian Generative Networks* — VAE-based Hamiltonian dynamics from pixel sequences.
- **pHNN (Desai 2021)** — *Port-Hamiltonian Neural Networks for Learning Explicit Time-Dependent Dynamical Systems* — adds dissipation + time-dependent forcing.
- **Adaptable HNN (Han 2021)** — *Adaptable Hamiltonian Neural Networks* — parameter-cognizant prediction across bifurcations / routes to chaos.
- **Self-Supervised HNN (Mattheakis 2022)** — *Hamiltonian Neural Networks for Solving Equations of Motion* — data-free NN solver for known Hamiltonians.
- **SHNN (David & Méhats 2023)** — *Symplectic Learning for Hamiltonian Neural Networks* — symplectic scheme in the loss; modified-Hamiltonian recovery.
- **sPHNN (Roth 2025)** — *Stable Port-Hamiltonian Neural Networks* — convex (FICNN) Hamiltonian giving global Lyapunov stability.
- **SNO (Makara 2026)** — *Symplectic Neural Operators* — structure-preserving operator learning for Hamiltonian PDEs.
- **GHNN (Horn 2025)** — *A Generalized Framework of Neural Networks for Hamiltonian Systems* — unifying framework benchmarked on gravitational N-body dynamics.
- **DHN (Deng 2026)** — *Dissipative Hamiltonian Neural Networks* — separates learning of conservative vs. dissipative dynamics.
- **ReviewHNNs (Chen 2022)** — *Learning Neural Hamiltonian Dynamics: A Methodological Overview* — survey/taxonomy of the HNN family.
