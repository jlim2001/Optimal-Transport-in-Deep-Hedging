"""
Penalized Wasserstein adversarial training for Heston deep hedging.

Derived from the authors' Heston_train_adv.py. Differences, all marked [FORK]:
  1. the constrained attack (SV_budget_attack, radius delta) is replaced by the
     penalized attack (penalty lam) from penalized.py;
  2. --lam, --iters, --kappa, --p0_attack, --b_max select the penalized adversary;
  3. --n_parts, --epochs, --warmup allow short smoke tests (defaults = authors' values);
  4. the realised radius of the adversary is logged each epoch, and every epoch is
     written to ../Result/<name>_part<K>_log.csv (see runlog.py).
Everything else (data, network, CVaR loss, optimiser, LR schedule, clean-loss weight
alpha1, BatchNorm momentum, p0 handling) is unchanged.
"""
import torch
import torch.nn as nn
from Heston_util import *
from penalized import Heston_Penalized_Attacker          # [FORK]
from runlog import EpochLogger                           # [FORK]
import argparse
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"running on {device}")


# learning rate schedule (unchanged from Heston_train_adv.py)
def alpha_learning_rate(epoch):
    if epoch < 200:
        return 1
    elif epoch < 500:
        return 0.1
    elif epoch < 600:
        return 0.01
    else:
        return 0.001


def epoch_pen_loader(loader, attacker, network, loss_fn, lam, alpha1, alpha2, opt=None):
    """One epoch; returns (adv loss, clean loss, mean realised radius)."""
    total_loss_att, total_loss_clean, total_radius = 0., 0., 0.
    for S, V, VarPrice in loader:
        S, V, VarPrice = S.to(device), V.to(device), VarPrice.to(device)
        network.train()
        input_vector = torch.cat((torch.log(S[:, :-1]).unsqueeze(-1), V[:, :-1].unsqueeze(-1)), dim=-1)
        holding = network(input_vector).squeeze()
        loss_clean = loss_fn(holding, S, VarPrice, p0=p0_clean)

        if alpha2 > 0:
            network.eval()
            # [FORK] penalized adversary instead of SV_budget_attack(network, S, V, delta, 4, 20)
            p0_for_attack = None if p0_attack == "search" else p0_att.item()
            S_att, V_att, VarPrice_att, info = attacker.penalized_attack(
                network, S, V, lam, n_iter=iters, attack_method=attack_method,
                kappa=kappa, p0=p0_for_attack, b_max=b_max)
            total_radius += info["radius"]
            input_vector_att = torch.cat((torch.log(S_att[:, :-1].clamp(min=1e-6)).unsqueeze(-1),
                                          V_att[:, :-1].unsqueeze(-1)), dim=-1)
            holding_att = network(input_vector_att).squeeze()
            # The penalty lam * c does not depend on theta, so (Danskin) the training
            # loss is simply the CVaR loss at the attacked paths, as in the authors' code.
            loss_att = loss_fn(holding_att, S_att, VarPrice_att, p0=p0_att)
        else:
            loss_att = torch.tensor([0.]).to(device)

        loss = alpha1 * loss_clean + alpha2 * loss_att
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total_loss_att += loss_att.item()
        total_loss_clean += loss_clean.item()
    n = len(loader)
    return total_loss_att / n, total_loss_clean / n, total_radius / n


sequence_length = 30
dt = 1/365
learning_rate = 0.005
batch_size = 10000
batch_num = 10

sigma = 2
T = dt * sequence_length
K = 100
s0 = 100
v0 = 0.04
alpha = 1.
b = 0.04
rho = -0.7
alpha_loss = 0.5

parser = argparse.ArgumentParser(description="Penalized Wasserstein adversarial training (Heston).")
parser.add_argument("--N", type=int, default=10000, help="number of samples.")
parser.add_argument("--lam", type=float, required=True, help="[FORK] penalty lambda (> 0).")
parser.add_argument("--alpha", type=float, default=1.0, help="weight on the CLEAN loss (authors' alpha).")
parser.add_argument("--attack_method", type=str, default="SV", help="attack method (S or SV).")
parser.add_argument("--transaction_cost_rate", type=float, default=0.0, help="transaction cost rate.")
parser.add_argument("--iters", type=int, default=20, help="[FORK] refinement steps of the penalized attack.")
parser.add_argument("--kappa", type=float, default=0.5, help="[FORK] budget-update damping in (0,1].")
parser.add_argument("--p0_attack", type=str, default="search", choices=["search", "given"],
                    help="[FORK] CVaR threshold inside the attack: batch quantile (as authors) or trained p0_att.")
parser.add_argument("--b_max", type=float, default=0.0, help="[FORK] cap on per-path budget (0 = none).")
parser.add_argument("--n_parts", type=int, default=0, help="[FORK] partitions to train (0 = all, as authors).")
parser.add_argument("--epochs", type=int, default=700, help="[FORK] total epochs (authors: 700).")
parser.add_argument("--warmup", type=int, default=300, help="[FORK] clean warm-up epochs (authors: 300).")
args = parser.parse_args()
print(vars(args))

N = args.N
lam = args.lam
alpha1 = args.alpha
alpha2 = 1.0
attack_method = args.attack_method
transaction_cost_rate = args.transaction_cost_rate
iters, kappa, p0_attack = args.iters, args.kappa, args.p0_attack
b_max = args.b_max if args.b_max > 0 else None
epoch_num, warmup = args.epochs, args.warmup

name = (f"Heston_{attack_method}pen_N{N:.0e}_lam{lam}_alpha{int(alpha1)}_tran{transaction_cost_rate:.0e}"
        .replace("+0", "").replace("+", ""))
Heston_data_train = torch.load('../Data/Heston_train.pt')

n_parts_total = int(1e5 / N)
n_parts = n_parts_total if args.n_parts <= 0 else min(args.n_parts, n_parts_total)

for part in range(0, n_parts):
    index_start = int(part * N)
    index_end = int((part + 1) * N)
    train_data = torch.utils.data.TensorDataset(Heston_data_train[0][index_start:index_end],
                                                Heston_data_train[1][index_start:index_end],
                                                Heston_data_train[2][index_start:index_end])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    network = RNN_BN_simple(sequence_length=sequence_length).to(device=device)
    loss_fn = loss_CVAR(Strike_price=K, vol=sigma, T=T, alpha_loss=alpha_loss,
                        trans_cost_rate=transaction_cost_rate, p0_mode='given').to(device=device)
    p0_clean = nn.Parameter(torch.tensor(1.69))
    p0_att = nn.Parameter(torch.tensor(1.69))
    opt = torch.optim.Adam([
        {'params': network.parameters()},
        {'params': [p0_clean, p0_att]}
    ], lr=learning_rate)
    LR_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=alpha_learning_rate)
    attacker = Heston_Penalized_Attacker(                                    # [FORK]
        loss_fn=loss_CVAR(Strike_price=K, vol=sigma, T=T, alpha_loss=alpha_loss,
                          trans_cost_rate=transaction_cost_rate, p0_mode='search').to(device=device),
        s0=s0, v0=v0, alpha=alpha, b=b, sigma=sigma, rho=rho, timestep=sequence_length, T=T)

    for module in network.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.momentum = 1.0

    print(f'Start Running {name}_part{int(part)}')
    # [FORK] per-epoch log next to the saved model (same part numbering as the .pth)
    logger = EpochLogger(f"../Result/{name}_part{int(part)+1}_log.csv")
    for i in range(epoch_num):
        time1 = time.time()
        network.train()
        if i < warmup:
            train_result = epoch_pen_loader(train_loader, attacker, network, loss_fn, lam, 1., 0., opt)
        else:
            if i == warmup:
                with torch.no_grad():
                    p0_att.copy_(p0_clean.detach().clone())
            train_result = epoch_pen_loader(train_loader, attacker, network, loss_fn, lam, alpha1, alpha2, opt)
        time2 = time.time()
        print(f"epoch{i},clean_loss: {train_result[1]:.6f}, att_loss: {train_result[0]:.6f}, "
              f"radius: {train_result[2]:.5f}, time: {time2-time1}s, "
              f"p0_clean: {p0_clean.item()}, p0_att: {p0_att.item()}")
        logger.log(epoch=i, phase="clean" if i < warmup else "robust",
                   clean_loss=train_result[1], att_loss=train_result[0],
                   radius=train_result[2] if i >= warmup else "",
                   seconds=time2 - time1, p0_clean=p0_clean.item(), p0_att=p0_att.item(),
                   lr=opt.param_groups[0]["lr"])
        LR_scheduler.step()

    logger.close()
    network.to('cpu')
    network.device = 'cpu'
    torch.save(network.state_dict(), f"../Result/{name}_part{int(part)+1}.pth")
