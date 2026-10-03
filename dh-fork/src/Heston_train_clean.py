import torch
import torch.nn as nn
from Heston_util import *
import argparse
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"running on {device}")



def alpha_learning_rate(epoch):
    if epoch<200:
        return 1
    elif epoch<400:
        return 0.1
    elif epoch<600:
        return 0.01
    else:
        return 0.001

def epoch_loader(loader, network, loss_fn, opt=None):
    """
    Runs one epoch of training or evaluation for clean (non-adversarial) training.

    Args:
        loader: DataLoader providing batches of price data.
        network: Neural network model.
        loss_fn: Loss function.
        opt: Optimizer (if training, None for evaluation).

    Returns:
        Average loss for the epoch.
    """
    total_loss = 0.
    for S,V ,VarPrice in loader:
        S, V, VarPrice = S.to(device), V.to(device), VarPrice.to(device)
        input_vector = torch.cat((torch.log(S[:, :-1]).unsqueeze(-1), V[:, :-1].unsqueeze(-1)), dim=-1)
        holding = network(input_vector).squeeze()
        loss = loss_fn(holding, S, VarPrice,p0 = p0_clean)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total_loss += loss.item()
    total_loss /= len(loader)
    return total_loss

sequence_length = 30
dt = 1/365
learning_rate = 0.05
batch_size = 10000
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


parser = argparse.ArgumentParser(description="Script for configuring network parameters.")

# Add arguments with default values
parser.add_argument("--N", type=int, default=10000, help="number of samples.")
parser.add_argument("--transaction_cost_rate", type=float, default=0.0, help="transaction cost rate.")
# Parse the arguments
args = parser.parse_args()
args_dict = vars(args)
print(args_dict)
N = args.N
transaction_cost_rate = args.transaction_cost_rate

name = f"HestonClean_N{N:.0e}_tran{transaction_cost_rate:.0e}".replace("+0", "").replace("-0", "-")
# Load training data
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
    opt = torch.optim.Adam([
        {'params': network.parameters()},  # Model parameters
        {'params': [p0_clean]}  # Trainable variable with its own learning rate
    ], lr=learning_rate)
    LR_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=alpha_learning_rate)
    attacker = Heston_Attacker(loss_fn=loss_CVAR(Strike_price=K, vol=sigma, T=T, alpha_loss=alpha_loss, trans_cost_rate=transaction_cost_rate, p0_mode='search').to(device=device),
                                s0=s0, v0=v0, alpha=alpha, b=b, sigma=sigma, rho=rho, timestep=sequence_length, T=T)

    # Training loop
    print(f'Start Running {name}_part{int(part)}')
    for i in range(epoch_num):
        time1 = time.time()
        network.train()
        train_result = epoch_loader(train_loader, network, loss_fn, opt=opt)
        time2 = time.time()
        print(f"epoch {i}, train loss: {train_result}, time: {time2-time1}")
        LR_scheduler.step()

    # Save trained network
    network.to('cpu')
    network.device = 'cpu'
    torch.save(network.state_dict(), f"../Result/{name}_part{int(part)+1}.pth")
