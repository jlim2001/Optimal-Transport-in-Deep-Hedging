import torch
import torch.nn as nn
from Heston_util import *
import argparse
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"running on {device}")


# learning rate schedule for alpha
def alpha_learning_rate(epoch):
    if epoch<200:
        return 1
    elif epoch<500:
        return 0.1
    elif epoch<600:
        return 0.01
    else:
        return 0.001


def epoch_att_loader(loader, attacker, network, loss_fn, delta,alpha1,alpha2, opt=None):
    """
    Runs one epoch of training or evaluation with adversarial attacks for GAD (Geometric Asian Delta) hedging.

    Args:
        loader: DataLoader providing batches of price data.
        attacker: Adversarial attacker object.
        network: Neural network model.
        loss_fn: Loss function.
        delta: Attack budget parameter.
        alpha1: Weight for clean loss.
        alpha2: Weight for adversarial loss.
        opt: Optimizer (if training, None for evaluation).

    Returns:
        Tuple of average clean loss and adversarial loss for the epoch.
    """
    total_loss_att, total_loss_clean = 0., 0.
    for S,V ,VarPrice in loader:
        S, V, VarPrice = S.to(device), V.to(device), VarPrice.to(device)
        network.train()
        input_vector = torch.cat((torch.log(S[:, :-1]).unsqueeze(-1), V[:, :-1].unsqueeze(-1)), dim=-1)
        holding = network(input_vector).squeeze()
        loss_clean = loss_fn(holding, S, VarPrice,p0 = p0_clean)

        if alpha2>0:
            network.eval()
            if attack_method == "S":
                S_att,V_att, VarPrice_att = attacker.S_budget_attack(network, S, V, delta, 4, 20)
            elif attack_method == "SV":
                S_att,V_att, VarPrice_att = attacker.SV_budget_attack(network, S, V, delta, 4, 20)
            input_vector_att = torch.cat((torch.log(S_att[:, :-1]).unsqueeze(-1), V_att[:, :-1].unsqueeze(-1)), dim=-1)
            holding_att = network(input_vector_att).squeeze()
            loss_att = loss_fn(holding_att, S_att, VarPrice_att,p0 = p0_att)
        else:
            loss_att = torch.tensor([0.]).to(device)
        
        loss = alpha1 * loss_clean + alpha2 * loss_att
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total_loss_att += loss_att.item()
        total_loss_clean += loss_clean.item()
    total_loss_att /= len(loader)
    total_loss_clean /= len(loader)
    return total_loss_att, total_loss_clean


sequence_length = 30
dt = 1/365
learning_rate = 0.005
batch_size = 10000
batch_num = 10
epoch_num = 700


sigma = 2
T = dt * sequence_length
K = 100
s0 = 100
v0 = 0.04
alpha = 1.
b = 0.04
rho = -0.7
alpha_loss = 0.5


# Create the parser
parser = argparse.ArgumentParser(description="Script for configuring network parameters.")

# Add arguments with default values
parser.add_argument("--N", type=int, default=10000, help="number of samples.")
parser.add_argument("--delta", type=float, default=0., help="attack delta.")
parser.add_argument("--alpha", type=float, default=1.0, help="alpha.")
parser.add_argument("--attack_method", type=str, default="S", help="attack method(S or SV).")
parser.add_argument("--transaction_cost_rate", type=float, default=0.0, help="transaction cost rate.")
# Parse the arguments
args = parser.parse_args()
args_dict = vars(args)
print(args_dict)
N = args.N
delta = args.delta
alpha1 = args.alpha
alpha2 = 1.0
attack_method = args.attack_method
transaction_cost_rate = args.transaction_cost_rate

name = f"Heston_{attack_method}att_N{N:.0e}_delta{delta}_alpha{int(alpha1)}_tran{transaction_cost_rate:.0e}".replace("+0", "").replace("+", "")
Heston_data_train = torch.load('../Data/Heston_train.pt')


# Loop over data partitions
for part in range(0, int(1e5/N)):
    index_start = int(part*N)
    index_end = int((part+1)*N)
    # Prepare partitioned training data and loader
    train_data = torch.utils.data.TensorDataset(Heston_data_train[0][index_start:index_end], Heston_data_train[1][index_start:index_end], Heston_data_train[2][index_start:index_end])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    network = RNN_BN_simple(sequence_length=sequence_length).to(device=device)
    # For Recurrent structure, replace above line by:
    # network = RNN_BN(sequence_length=sequence_length).to(device=device)
    loss_fn = loss_CVAR(Strike_price=K, vol=sigma, T=T, alpha_loss=alpha_loss, trans_cost_rate=transaction_cost_rate, p0_mode='given').to(device=device)
    p0_clean = nn.Parameter(torch.tensor(1.69))
    p0_att = nn.Parameter(torch.tensor(1.69))
    opt = torch.optim.Adam([
        {'params': network.parameters()},  # Model parameters
        {'params': [p0_clean, p0_att]}  # Trainable variable with its own learning rate
    ], lr=learning_rate)
    LR_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=alpha_learning_rate)
    attacker = Heston_Attacker(loss_fn=loss_CVAR(Strike_price=K, vol=sigma, T=T, alpha_loss=alpha_loss, trans_cost_rate=transaction_cost_rate, p0_mode='search').to(device=device),
                                s0=s0, v0=v0, alpha=alpha, b=b, sigma=sigma, rho=rho, timestep=sequence_length, T=T)

    for module in network.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.momentum = 1.0

    print(f'Start Running {name}_part{int(part)}')
    # Training loop
    for i in range(epoch_num):
        time1 = time.time()
        network.train()
        if i < 300:
            # Train with only clean loss for first 300 epochs
            train_result = epoch_att_loader(train_loader,attacker, network, loss_fn,delta,1.,0., opt)
        else:
            if i == 300:
                # Synchronize p0_att with p0_clean at epoch 300
                with torch.no_grad():
                    p0_att.copy_(p0_clean.detach().clone())
            # Train with balanced clean and adversarial loss
            train_result = epoch_att_loader(train_loader,attacker, network, loss_fn,delta,alpha1,alpha2, opt)
        
        time2 = time.time()
        print(f"epoch{i},clean_loss: {train_result[1]:.6f}, att_loss: {train_result[0]:.6f},time: {time2-time1}s, p0_clean: {p0_clean.item()}, p0_att: {p0_att.item()}")
        LR_scheduler.step()

    # Save trained network
    network.to('cpu')
    network.device = 'cpu'
    torch.save(network.state_dict(), f"../Result/{name}_part{int(part)+1}.pth")

