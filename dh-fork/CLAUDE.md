# CLAUDE.md — project context

## Goal
He, Sutter & Gonon (2025), *Distributional Adversarial Attacks and Training in Deep
Hedging*, make deep hedging robust via adversarial training: each batch runs a
20-iteration WBPGD attack on the price/variance paths. That costs ~10x clean training.

**Our contribution:** replace the attack with the first-order Wasserstein sensitivity
expansion (Bartl, Drapeau, Obłój & Wiesel, *Sensitivity analysis of Wasserstein DRO*):

    loss_att = l(x) + delta * Upsilon(theta)

and measure (a) how much out-of-distribution robustness survives, (b) the compute saved,
(c) the radius delta at which the first-order approximation breaks down.
Secondary: cheap training makes exploring delta practical — the authors name sensitivity
to delta as a limitation. (Choosing delta still needs validation; Upsilon itself is
independent of delta, the trained network is not.)

## Scope decisions (settled)
- Heston only (Black–Scholes dropped). SV-attack for the headline comparison.
- Keep their CVaR loss (level 0.5, trained threshold p0). No entropic loss.
- Baseline code = this repo (fork of the authors'). Minimal changes only.

## What we changed (everything else is the authors' code, unchanged)
- `src/sensitivity.py` — NEW. `upsilon(...)` computes
  `sqrt(mean_n sum_{c in S,V} ||grad_{x_nc} l_n||_1^2)` (l1 over t=1..T), matching the
  ball searched by `Heston_Attacker.SV_budget_attack`: per-path sup-norm over time,
  W2-type budget across paths, V perturbed in units of 1/100, t=0 never perturbed,
  VarPrice recomputed from perturbed V. Computed in eval mode (BatchNorm) with
  create_graph=True. Verified numerically (NumPy, linear loss) to equal the attack's
  worst case exactly.
- `src/Heston_train_sens.py` — copy of `Heston_train_adv.py` with 3 edits: import,
  attack branch -> `loss_att = loss_x + delta * ups`, output prefix `att` -> `sens`.

## Facts about the authors' code (some differ from the paper text)
- `--alpha` weights the CLEAN loss: `loss = alpha*clean + 1.0*adversarial`.
- Heston: S0=K=100, v0=0.04, kappa=1, b=0.04, vol-of-vol sigma=2, rho=-0.7,
  30 daily steps, T=30/365. Network: 30 per-step nets (RNN_BN_simple), input (log S, V).
- LR: clean script 0.05 (decay at 200/400/600); adv script 0.005 (200/500/600).
  Sens inherits adv settings. Matched clean control: `Heston_train_sens.py --delta 0`.
- Each run trains 100000/N networks on disjoint partitions -> cost ~ one N=100k run
  (~10h adversarial on CPU) regardless of N.
- Table 5a SV-attack (delta, alpha): 5k (0.5,1), 10k (1.0,10), 20k (0.1,1),
  50k (0.03,0), 100k (0.005,0).
- S-attack caveat: S_budget_attack's effective S-radius is between delta/sqrt(2) and delta.

## Status
- NOTHING HAS BEEN RUN. Code was written in an environment without working PyTorch.
- Riskiest untested part: double backprop for Upsilon through the 30 per-step nets.

## Next steps (in order)
1. Generate data: `cd src && python Heston_generator.py`.
2. Add CLI flags to `Heston_train_sens.py` (and adv): `--epochs`, `--warmup`,
   `--n_parts` (replace `range(0, int(1e5/N))` with a capped range). Smoke-test all
   three scripts with ~2 epochs, 1 partition.
3. Write `src/breakdown_check.py` (no training): load `Result/Heston.pth`, compute
   l and Upsilon once, run `SV_budget_attack` at delta in {0.001..1.0}, plot attacked
   loss vs l + delta*Upsilon. Curves must agree as delta -> 0 (sanity check); where
   they separate = breakdown radius. Unknown which run produced Heston.pth.
4. Main comparison: clean / adv / sens, SV-attack, N in {5k, 20k, 100k}, Table 5
   (delta, alpha), ~3 partitions each. Collect per-epoch wall-clock for cost.
   Evaluate in-distribution (Heston_test.pt) and OOD (Heston_OODP.pt, ±10% params).
5. Delta sweep with sens only at one N.

## Write-up caveats
First-order is exact only as delta -> 0 (Table 5 uses delta up to 1.0); CVaR kink
handled a.e. by autodiff; S-attack radius ambiguity; LR inconsistency in authors' code.
