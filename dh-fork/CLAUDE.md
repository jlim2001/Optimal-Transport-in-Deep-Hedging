# CLAUDE.md — project context (read first)

## Research question
He, Sutter & Gonon (NeurIPS 2025) train deep hedges adversarially against the worst case in a
Wasserstein ball of FIXED radius delta (hard constraint) and report that performance is
sensitive to delta. We replace the constraint with a Lagrangian PENALTY lam (penalized
Wasserstein DRO, Sinha–Namkoong–Duchi 2018), same transport cost, and compare:
 (1) hedging risk in- and out-of-distribution at matched strength;
 (2) sensitivity of performance to the hyperparameter (lam vs delta) — the headline;
 (3) cross-robustness: each trained model attacked by both adversaries.
Cost/speed is NOT the research question (report it only as context).
An earlier direction (first-order sensitivity penalty as a cheap replacement) was dropped;
`sensitivity.py` survives only to calibrate lam.

## Settled scope
- Heston only, SV-attack for headline results, authors' CVaR loss (level 0.5) unchanged.
- This repo = fork of the authors' code. ORIGINAL FILES ARE UNMODIFIED; we only add files.
- Change one thing at a time: same geometry as the authors (per-path sup-norm over time,
  squared and summed over S,V; V in units of 1/100; t=0 fixed); only constraint -> penalty.
  A different cost (e.g. squared l2) would be a separate, secondary experiment.

## Our files (src/)
- `penalized.py` — `Heston_Penalized_Attacker.penalized_attack(network,S,V,lam,n_iter,...)`:
  linearised init (s=sign g, b=||g||_1/(2 lam)), then n_iter refinements (authors' sign
  update; damped budget best response), keep best per path. Returns info["radius"].
- `Heston_train_pen.py` — copy of `Heston_train_adv.py` with the attack swapped; flags
  --lam --iters --kappa --p0_attack --b_max --n_parts --epochs --warmup (defaults = authors').
- `calibrate_lambda.py` — lam(delta) = Upsilon/(2 delta) from a clean network; --verify runs
  the attack and prints realised radius / delta.
- `evaluate.py` — in-dist CVaR (Heston_test.pt), OOD mean/worst over 100 configs
  (Heston_OODP.pt), CVaR under constrained (deltas) and penalized (lams) attacks with radii -> CSV.
- `sensitivity.py` — Upsilon for the authors' ball (verified numerically).
- `naming.py` (parse run names), `runlog.py` (per-epoch log format), `parse_logs.py`
  (authors' printed output -> *_log.csv), `summarise.py` (tables + figures -> Result/summary/).
  Results pipeline is documented in README_FORK.md 'Where results are saved'.
  ALWAYS run the authors' scripts with `2>&1 | tee ../Result/stdout_<run>.txt` (they don't log).

## Facts about the authors' code
- `--alpha` weights the CLEAN loss: loss = alpha*clean + 1.0*adversarial.
- Heston S0=K=100, v0=0.04, kappa=1, b=0.04, vol-of-vol=2, rho=-0.7, drift 0, 30 steps, T=30/365.
- LR: clean script 0.05 (200/400/600); adv script 0.005 (200/500/600); pen inherits adv.
- SV_budget_attack always runs 20 iterations (iter arg only sets step size).
- Each run trains 100000/N networks; cost per run ~ one N=1e5 run (~10h CPU adversarial).
- Model filenames: Heston_SVatt_N1e4_delta1.0_alpha10_tran0e0_partK.pth,
  Heston_SVpen_N1e4_lam12.5_alpha10_tran0e0_partK.pth.
- Table 5a SV (delta, alpha): 5k(0.5,1) 10k(1.0,10) 20k(0.1,1) 50k(0.03,0) 100k(0.005,0).

## Status
- NOTHING HAS BEEN RUN (written without working PyTorch). Python compiles; attacker math
  verified in NumPy (linear-loss optimum, radius = Ups/(2 lam), convergence iff lam > -mu/2,
  divergence below it).
- Riskiest untested parts: penalized_attack autograd on leaves b, s; per-path best tracking;
  double backprop in sensitivity.upsilon (calibration only).

## Next steps
1. `python Heston_generator.py`; smoke test:
   `python Heston_train_pen.py --N 10000 --lam 10 --n_parts 1 --epochs 3 --warmup 1 --iters 2`.
2. Train a clean network: `python Heston_train_adv.py --N 100000 --delta 0 --alpha 1 --attack_method SV`
   (or with fewer epochs first). Run `calibrate_lambda.py --verify` on it.
3. Matched comparison at N in {10k, 20k} (then others): adv with Table-5 delta; pen with
   calibrated lam; --n_parts 3. Evaluate with evaluate.py incl. cross-attacks.
4. Hyperparameter sweep at one N: adv over delta grid {0.005..1.0}, pen over lam grid around
   the calibrated values. Plot OOD CVaR vs REALISED radius for both (common axis).
5. Watch for: exploding radius at small lam (use --b_max), radius drift during training,
   NaNs from log of non-positive S (pen script clamps at 1e-6).

## Write-up
LaTeX skeleton in `report/report.tex` (guidance boxes say what goes where).
