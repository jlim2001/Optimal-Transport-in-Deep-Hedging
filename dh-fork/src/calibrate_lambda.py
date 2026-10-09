"""
Calibrate the penalty lambda to the constrained radius delta.

First-order theory (Bartl, Drapeau, Oblój & Wiesel 2021, applied to this geometry):
for the penalized adversary, the per-path best response to the linearised loss is
b_{n,c} = ||g_{n,c}||_1 / (2 lam), so the realised radius is
        sqrt(mean_n sum_c b_{n,c}^2)  =  Upsilon / (2 lam)
where Upsilon is the sensitivity term computed in sensitivity.py for the SAME cost.
Setting this equal to delta gives
        lam(delta)  =  Upsilon / (2 delta).

This script: (1) loads a trained network, (2) computes Upsilon on validation paths,
(3) prints lam(delta) for each delta, and (4) with --verify, runs the full penalized
attack at that lam and reports the realised radius, so you can see how far the
nonlinear attack drifts from the first-order prediction.

Calibrate with the network the robust phase STARTS from (a clean network), since
training begins from the clean warm-up. Upsilon shrinks as a network becomes robust,
so with a fixed lam the realised radius typically shrinks during training (logged by
Heston_train_pen.py).

Example:
    python calibrate_lambda.py --model ../Result/Heston_SVatt_N1e5_delta0.0_alpha1_tran0e0_part1.pth --verify
"""
import argparse
import torch
import torch.nn as nn
from Heston_util import *
from sensitivity import upsilon
from penalized import Heston_Penalized_Attacker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sequence_length, dt = 30, 1/365
sigma, T, K, s0, v0, alpha, b, rho, alpha_loss = 2, 30/365, 100, 100, 0.04, 1., 0.04, -0.7, 0.5
TABLE5_SV = [0.5, 1.0, 0.1, 0.03, 0.005]          # He-Sutter-Gonon Table 5a, SV-attack

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="../Result/Heston.pth")
parser.add_argument("--data", type=str, default="../Data/Heston_val.pt")
parser.add_argument("--n_paths", type=int, default=10000)
parser.add_argument("--attack_method", type=str, default="SV")
parser.add_argument("--deltas", type=float, nargs="*", default=sorted(set(TABLE5_SV)))
parser.add_argument("--verify", action="store_true", help="run the penalized attack at each lam")
parser.add_argument("--iters", type=int, default=20)
args = parser.parse_args()

network = RNN_BN_simple(sequence_length=sequence_length).to(device)
network.load_state_dict(torch.load(args.model, map_location=device))
network.eval()

data = torch.load(args.data)
S, V = data[0][:args.n_paths].to(device), data[1][:args.n_paths].to(device)

attacker = Heston_Penalized_Attacker(
    loss_fn=loss_CVAR(Strike_price=K, vol=sigma, T=T, alpha_loss=alpha_loss, p0_mode='search').to(device),
    s0=s0, v0=v0, alpha=alpha, b=b, sigma=sigma, rho=rho, timestep=sequence_length, T=T)

# threshold p0 at the clean paths (batch quantile), then Upsilon with that threshold
clean_cvar, _, p0 = attacker.perfromance(network, S, V)
loss_given = loss_CVAR(Strike_price=K, vol=sigma, T=T, alpha_loss=alpha_loss, p0_mode='given').to(device)
ups, _ = upsilon(network, loss_given, attacker, S, V, p0,
                 attack_method=args.attack_method, create_graph=False)
ups = ups.item()

print(f"model: {args.model}")
print(f"clean CVaR = {clean_cvar:.5f}   p0 = {p0:.5f}   Upsilon = {ups:.5f}\n")
header = f"{'delta':>8} {'lambda=Ups/(2delta)':>20}"
if args.verify:
    header += f" {'realised radius':>16} {'radius/delta':>13} {'attacked CVaR':>14}"
print(header)
for d in args.deltas:
    lam = ups / (2.0 * d)
    row = f"{d:8.4f} {lam:20.5f}"
    if args.verify:
        S_a, V_a, VP_a, info = attacker.penalized_attack(network, S, V, lam, n_iter=args.iters,
                                                         attack_method=args.attack_method)
        att_cvar = attacker.perfromance(network, S_a, V_a)[0]
        row += f" {info['radius']:16.5f} {info['radius']/d:13.3f} {att_cvar:14.5f}"
    print(row)
print("\nradius/delta near 1 at small delta = calibration consistent with first-order theory;")
print("drift from 1 at large delta = nonlinearity of the loss in the paths.")
