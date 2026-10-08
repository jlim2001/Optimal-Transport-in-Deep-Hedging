"""
Evaluate trained hedging networks (clean, constrained-adversarial, penalized) on:
  * in-distribution CVaR       : Heston_test.pt (first --n_test paths)
  * out-of-distribution CVaR   : Heston_OODP.pt = 100 parameter configs x 10,000 paths
                                 (each Heston parameter scaled by up to +/-10%);
                                 reports the mean and the worst config
  * attacked CVaR, both ways   : constrained attack at each --deltas (authors' SV/S budget
                                 attack) and penalized attack at each --lams, on --n_attack
                                 test paths, with the realised radius of each attack
                                 (cross-evaluation: every model faces both adversaries)

Usage:
    python evaluate.py --models "../Result/Heston_SVatt_N1e4*" "../Result/Heston_SVpen_N1e4*" \
                       --deltas 0.1 1.0 --lams 5 50 --out ../Result/eval.csv
Rows are written per model file (= per partition); average over partitions downstream.
"""
import argparse
import csv
import glob
import torch
from Heston_util import *
from penalized import Heston_Penalized_Attacker, realised_radius_from_att

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sequence_length = 30
sigma, T, K, s0, v0, alpha, b, rho, alpha_loss = 2, 30/365, 100, 100, 0.04, 1., 0.04, -0.7, 0.5

parser = argparse.ArgumentParser()
parser.add_argument("--models", nargs="+", required=True, help="model files or glob patterns")
parser.add_argument("--n_test", type=int, default=100000)
parser.add_argument("--n_attack", type=int, default=10000)
parser.add_argument("--attack_method", type=str, default="SV")
parser.add_argument("--deltas", type=float, nargs="*", default=[])
parser.add_argument("--lams", type=float, nargs="*", default=[])
parser.add_argument("--iters", type=int, default=20)
parser.add_argument("--out", type=str, default="../Result/eval.csv")
args = parser.parse_args()

files = sorted({f for pat in args.models for f in glob.glob(pat)})
if not files:
    raise SystemExit("no model files matched")

attacker = Heston_Penalized_Attacker(
    loss_fn=loss_CVAR(Strike_price=K, vol=sigma, T=T, alpha_loss=alpha_loss, p0_mode='search').to(device),
    s0=s0, v0=v0, alpha=alpha, b=b, sigma=sigma, rho=rho, timestep=sequence_length, T=T)

test = torch.load('../Data/Heston_test.pt')
S_te, V_te = test[0][:args.n_test].to(device), test[1][:args.n_test].to(device)
S_at, V_at = S_te[:args.n_attack], V_te[:args.n_attack]
oodp = torch.load('../Data/Heston_OODP.pt')
n_cfg, per_cfg = 100, 10000

rows = []
for f in files:
    net = RNN_BN_simple(sequence_length=sequence_length).to(device)
    net.load_state_dict(torch.load(f, map_location=device))
    net.eval()
    row = {"model": f}
    with torch.no_grad():
        row["cvar_in"] = attacker.perfromance(net, S_te, V_te)[0]
        ood = []
        for k in range(n_cfg):
            sl = slice(k * per_cfg, (k + 1) * per_cfg)
            ood.append(attacker.perfromance(net, oodp[0][sl].to(device), oodp[1][sl].to(device))[0])
    row["cvar_ood_mean"] = sum(ood) / len(ood)
    row["cvar_ood_worst"] = max(ood)

    for d in args.deltas:
        att = (attacker.SV_budget_attack if args.attack_method == "SV" else attacker.S_budget_attack)
        S_a, V_a, _, a = att(net, S_at, V_at, d, 4, 20, return_att=True)
        row[f"cvar_constrained_d{d}"] = attacker.perfromance(net, S_a, V_a)[0]
        # S-attack returns a 2-column att whose V column is never applied -> use S column only
        a_used = a if args.attack_method == "SV" else a[:, :, :1]
        row[f"radius_constrained_d{d}"] = realised_radius_from_att(a_used)
    for lam in args.lams:
        S_a, V_a, _, info = attacker.penalized_attack(net, S_at, V_at, lam, n_iter=args.iters,
                                                      attack_method=args.attack_method)
        row[f"cvar_penalized_l{lam}"] = attacker.perfromance(net, S_a, V_a)[0]
        row[f"radius_penalized_l{lam}"] = info["radius"]
    rows.append(row)
    print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})

with open(args.out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {len(rows)} rows to {args.out}")
