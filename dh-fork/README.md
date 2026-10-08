# Fork: penalized vs. constrained Wasserstein adversarial training in deep hedging

Fork of He, Sutter & Gonon, *Distributional Adversarial Attacks and Training in Deep Hedging*
(NeurIPS 2025; MIT licence, original notice kept in `LICENSE`).
Original repo: https://github.com/Guangyi-Mira/Distributional-Adversarial-Attacks-and-Training-in-Deep-Hedging

## Research question

He–Sutter–Gonon train hedges against the worst case in a Wasserstein ball of **fixed radius δ**
(a hard constraint), and note that performance is sensitive to the choice of δ. We replace the
hard constraint with a **Lagrangian penalty λ** (penalized Wasserstein DRO, Sinha, Namkoong &
Duchi 2018), keeping the same transport cost, and ask:

1. At matched strength, does penalized training hedge as well in- and out-of-distribution?
2. Is it **less sensitive to its hyperparameter** (λ) than constrained training is to δ?
3. Does robustness learned under one adversary transfer to the other?

| | Constrained (authors) | Penalized (this fork) |
|---|---|---|
| Worst case over | distributions within W-distance δ | all distributions, minus λ × transport cost |
| Inner problem | one shared budget over all paths, projected | separate maximisation per path, no projection |
| Hyperparameter | radius δ | penalty λ (radius becomes implicit and adaptive) |
| Script | `Heston_train_adv.py` (unchanged) | `Heston_train_pen.py` (new) |

## What we added (no original file was modified)

| file | purpose |
|---|---|
| `src/penalized.py` | penalized attacker; same budget × sign parameterisation and cost as `SV_budget_attack` |
| `src/Heston_train_pen.py` | training with the penalized attacker; otherwise identical to `Heston_train_adv.py` |
| `src/calibrate_lambda.py` | maps each δ to a matched λ ≈ Υ/(2δ) and verifies the realised radius |
| `src/evaluate.py` | in-distribution, out-of-distribution and cross-attack CVaR → CSV |
| `src/sensitivity.py` | the first-order term Υ (Bartl–Drapeau–Obłój–Wiesel), used for calibration |

### Same geometry, different enforcement
Perturbation of path n, coordinate c ∈ {S, V}: `D = b_{n,c} · s_{n,c,t}`, `b ≥ 0`, `s ∈ [−1, 1]`;
cost `c_n = Σ_c max_t |D_{n,c,t}|²`; V perturbed in units of 1/100; t = 0 never perturbed;
variance-swap price recomputed from the perturbed V. The constrained attack enforces
`mean_n c_n ≤ δ²`; the penalized attack maximises `l_n − λ c_n` for each path.

### The penalized attack
1. **Initialise** at the exact maximiser of the linearised problem: `s = sign(g)`, `b_c = ‖g_c‖₁ / (2λ)`.
2. **Refine** (`--iters`, default 20): the authors' signed-gradient update for `s`; a damped
   best response for the budget, `b ← (1−κ) b + κ (∂l/∂b) / (2λ m²)`, `m = max_t |s|`.
3. **Keep the best iterate per path** by penalised objective.

### Calibration λ ↔ δ
To first order the realised radius is `Υ / (2λ)`, so **λ(δ) = Υ / (2δ)** with Υ from
`sensitivity.py`. Calibrate on a clean network (the robust phase starts from the clean warm-up).
Υ shrinks as the network becomes robust, so with fixed λ the realised radius typically shrinks
during training; `Heston_train_pen.py` logs it every epoch.

### Verified (NumPy, without PyTorch)
* Linear loss: the initialisation is optimal, the objective equals Υ²/(4λ), the radius equals
  Υ/(2λ), λ = Υ/(2δ) gives radius δ exactly, and random perturbations never beat it.
* Loss curved in the paths, `l(b) = Gb − (μ/2)b²`: the budget update converges to `G/(μ+2λ)`
  whenever **λ > −μ/2**; below that threshold the inner problem is unbounded (budgets diverge).
  Near the threshold convergence is slow (contraction factor `1 − κ(1 + μ/(2λ))` per step).

**Not yet run end-to-end** — PyTorch was unavailable where this was written. Smoke-test first.

## Facts about the authors' code
- `--alpha` weights the **clean** loss: `loss = alpha·clean + 1.0·adversarial`.
- CVaR level 0.5; threshold `p0` trained (`p0_clean`, `p0_att`, init 1.69). Inside the attack the
  threshold is the batch quantile (`p0_mode='search'`). `--p0_attack given` uses `p0_att` instead.
- Heston: S0 = K = 100, v0 = 0.04, κ = 1, b = 0.04, vol-of-vol σ = 2, ρ = −0.7, drift 0,
  30 daily steps, T = 30/365. One network per time step (`RNN_BN_simple`), input (log S, V).
- Learning rates differ between scripts: `Heston_train_clean.py` 0.05 (decay at 200/400/600),
  `Heston_train_adv.py` 0.005 (decay at 200/500/600). `Heston_train_pen.py` inherits the adv
  settings, so constrained vs. penalized is a matched comparison.
- `SV_budget_attack` always runs 20 iterations; its `iter` argument only sets the step size.
- `S_budget_attack` keeps an unused V budget column in its normalisation (effective S-radius
  between δ/√2 and δ). Use the SV-attack for headline results.
- Each run trains `100000/N` networks on disjoint partitions; `--n_parts` caps this (pen script).
- Out-of-distribution set: 100 parameter configurations × 10,000 paths (κ, b, σ, ρ each scaled
  by up to ±10%).

## Run

```bash
cd src
python Heston_generator.py                       # ../Data/Heston_{train,val,test,OODP}.pt

# smoke test (minutes)
python Heston_train_pen.py --N 10000 --lam 10 --n_parts 1 --epochs 3 --warmup 1 --iters 2

# 1. clean network for calibration (authors' script, delta 0 = no attack)
python Heston_train_adv.py --N 100000 --delta 0 --alpha 1 --attack_method SV
# 2. matched lambda for each Table-5 delta
python calibrate_lambda.py --model ../Result/Heston_SVatt_N1e5_delta0.0_alpha1_tran0e0_part1.pth --verify
# 3. constrained (authors) and penalized (ours) at matched strength, e.g. N = 10k
python Heston_train_adv.py --N 10000 --delta 1.0 --alpha 10 --attack_method SV
python Heston_train_pen.py --N 10000 --lam <from step 2> --alpha 10 --attack_method SV --n_parts 3
# 4. evaluate everything, including cross-attacks
python evaluate.py --models "../Result/Heston_SV*_N1e4*" --deltas 0.1 1.0 --lams <lams> --out ../Result/eval_N1e4.csv
```

Table 5a (SV-attack) (δ, alpha): 5k (0.5, 1) · 10k (1.0, 10) · 20k (0.1, 1) · 50k (0.03, 0) · 100k (0.005, 0).

## Caveats
1. **No convergence guarantee.** Sinha et al.'s guarantee needs a loss smooth in the paths and
   λ above its curvature; ReLU, the CVaR kink and the sup-norm cost all break smoothness. Check
   that the logged radius and attacked loss are stable, and report the iteration count.
2. **Small λ runs away** (see verification above). Use `--b_max` or larger λ if radii explode.
3. **Threshold choice.** Default `--p0_attack search` mirrors the authors' attack; `given` makes
   the inner problem exactly separable per path. Report which one you use.
4. **Calibration is first order**: expect realised radius / δ to drift from 1 at large δ.
