# Fork: first-order sensitivity training vs. WBPGD adversarial training

Fork of He, Sutter & Gonon (2025), *Distributional Adversarial Attacks and Training
in Deep Hedging* (MIT licence — original notice kept in `LICENSE`).
Original repo: https://github.com/Guangyi-Mira/Distributional-Adversarial-Attacks-and-Training-in-Deep-Hedging

## The one change

| file | status |
|---|---|
| `src/sensitivity.py` | **new** — the first-order penalty Υ |
| `src/Heston_train_sens.py` | **new** — copy of `Heston_train_adv.py` with only the attack branch replaced |
| everything else | **unchanged** (data generation, network, CVaR loss, schedule, evaluation) |

In `Heston_train_adv.py`, each batch runs a 20-iteration WBPGD attack and evaluates the
loss at the attacked paths. `Heston_train_sens.py` replaces that with the first-order
expansion of the worst case (Bartl–Drapeau–Obłój–Wiesel):

    loss_att  =  l(x)  +  delta * Upsilon(theta)

so one extra (double-)backward pass replaces 20 attack iterations. Full diff vs. their script:
three edits (import, attack branch, output filename prefix `att` → `sens`).

### Υ matches their attack's Wasserstein ball

Read from `Heston_Attacker.SV_budget_attack`: per-path sup-norm over time, W2-type
budget across paths (`mean_n Σ_c ||Δ_nc||_∞² ≤ δ²`), V perturbed in units of 1/100,
t = 0 never perturbed, variance-swap price recomputed from perturbed V. The dual is

    Upsilon = sqrt( mean_n  Σ_{c∈{S,V}}  || ∇_{x_nc} l_n ||_1² )   (l1 over t = 1..T)

Checked numerically: for a loss linear in the paths (first order exact), the authors'
attack structure and δ·Υ give identical worst-case values, and random feasible
perturbations never exceed δ·Υ.

## What their code actually does (differs from the paper text in places)

- **Risk measure:** CVaR at level 0.5 (`alpha_loss = 0.5`), Rockafellar–Uryasev form;
  the threshold `p0` is **trained**, with separate `p0_clean` and `p0_att` (init 1.69).
- **`--alpha` weights the CLEAN loss:** `loss = alpha * clean + 1.0 * adversarial`.
  So in Table 5, α = 0 means the robust phase trains on the adversarial loss only.
- **Heston:** S0 = K = 100, v0 = 0.04, κ = 1, b = 0.04, **vol-of-vol σ = 2**, ρ = −0.7,
  30 daily steps, T = 30/365.
- **Optimiser:** Adam, batch 10,000, 700 epochs (first 300 clean in the adv script).
  Learning rates are NOT consistent across their Heston scripts:
  `Heston_train_clean.py` uses lr 0.05 (matches the paper), decay ×0.1 at 200/400/600;
  `Heston_train_adv.py` uses lr 0.005, decay ×0.1 at 200/500/600.
  `Heston_train_sens.py` inherits the adv settings, so sens vs. adv is matched.
  For a matched clean control, run `Heston_train_sens.py --delta 0`.
- **Network:** 30 separate per-time-step nets (`RNN_BN_simple`), input (log S, V),
  output 2 holdings (stock, variance swap), BatchNorm.
- **Partitions:** each run trains `100000 / N` independent networks on disjoint subsets
  (N = 5,000 → 20 networks). Budget runtime accordingly.

## Run

```bash
cd src
python Heston_generator.py                                   # writes ../Data/Heston_*.pt

# baseline (clean)
python Heston_train_clean.py --N 10000 --transaction_cost_rate 0.0
# their robust method (expensive)
python Heston_train_adv.py  --N 10000 --delta 1.0 --alpha 10 --attack_method SV --transaction_cost_rate 0.0
# ours (cheap)
python Heston_train_sens.py --N 10000 --delta 1.0 --alpha 10 --attack_method SV --transaction_cost_rate 0.0
```

(δ, α) per N for SV-Attack from Table 5a: 5k (0.5, 1) · 10k (1.0, 10) · 20k (0.1, 1) ·
50k (0.03, 0) · 100k (0.005, 0).

**Cost comparison:** both training scripts print wall-clock time per epoch — compare the
robust-phase epochs (300+) of `adv` vs. `sens`.

**Evaluation:** in `Heston_evaluation.ipynb`, change the `load_state_dict` filename to the
`..._SVsens_...partK.pth` file (and the `..._SVatt_...` file for the baseline). Use the
in-distribution test set and `Heston_OODP.pt` (±10% parameter shifts) for robustness.

## Caveats to state in the write-up

1. **First order vs. large δ.** Table 5 uses δ up to 1.0 for SV at small N. The expansion
   is only exact as δ → 0, so where `sens` falls short of `adv` at large δ is itself a
   result (the breakdown radius), not a bug. Sweep δ to find it.
2. **S-attack radius.** `S_budget_attack` keeps an unused second budget column in its
   normalisation, so its effective S-radius lies between δ/√2 and δ. Υ uses δ. Prefer
   SV-Attack for the headline comparison, where the geometry matches exactly.
3. **CVaR kink.** Υ uses gradients of max(X − p0, 0); these exist almost everywhere
   (same as the ReLU kinks already in the network), so autodiff handles it. Paths
   sitting exactly at the threshold contribute zero gradient.
4. **BatchNorm mode.** Υ is computed in eval mode (as the authors do for the attack
   branch) so per-path gradients don't depend on the rest of the batch.
5. **Not yet executed end-to-end** — run a 1-epoch smoke test first
   (e.g. temporarily set `epoch_num = 2` and the warm-up threshold `300 → 1`).
