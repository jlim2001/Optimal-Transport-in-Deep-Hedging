#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
from GAD_util import *
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"running on {device}")


def epoch_att_loader(loader,attacker, network, loss_fn, delta, alpha1=1., alpha2=1,opt=None):
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
    total_loss_clean,total_loss_att = 0.,0.
    for price, in loader:
        price  = price.to(device)
        network.train()
        input_tensor = (price[:,:-1].unsqueeze(-1))
        holding = network(input_tensor).squeeze()
        loss_clean = loss_fn(holding, price, p0_clean)

        if alpha2>0:
            network.eval()
            att,_ = attacker.budget_att(network, price, delta, 4, 20)
            price_att = price + att.to(device)
            input_tensor_att = (price_att[:,:-1].unsqueeze(-1))    
            holding_att = network(input_tensor_att).squeeze()
            loss_att = loss_fn(holding_att, price_att,p0_att)
        else:
            loss_att = torch.tensor([0.]).to(device)

        loss = alpha1*loss_clean + alpha2*loss_att
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total_loss_clean += loss_clean.item()
        total_loss_att += loss_att.item()
    total_loss_clean /= len(loader)
    total_loss_att /= len(loader)
    return total_loss_clean, total_loss_att

# learning rate schedule for alpha
def alpha_learning_rate(epoch):
    if epoch<100:
        return 1
    elif epoch<200:
        return 0.1
    elif epoch<250:
        return 0.01
    else:
        return 0.001


# Define parameters for the GAD model
sequence_length = 30
dt = 1/250  # Daily trading frequency (250 trading days per year)
learning_rate = 0.05
epoch_num = 300

# Create the parser
parser = argparse.ArgumentParser(description="Script for configuring network parameters.")

# Add arguments with default values
parser.add_argument("--N", type=int, default=10000, help="number of data.")
parser.add_argument("--company", type=str, default='AAPL', help="company name.")
parser.add_argument("--input", type=str, default='fix', help="input dataset(fix or robust)")
parser.add_argument("--delta", type=float, default=0.1, help="attack delta.")
parser.add_argument("--alpha", type=float, default=1.0, help="alpha.")
# Parse the arguments
args = parser.parse_args()
args_dict = vars(args)
print(args_dict)

# Extract parsed arguments
N = args.N
input_dataset = args.input
delta = args.delta
alpha1 = args.alpha
alpha2 = 1.
company_name = args.company

# Define the name for saving the model
name = f"GAD_{company_name}_{input_dataset}_N{N:.0e}_delta{delta}_alpha_{int(alpha1)}_{int(alpha2)}".replace("+0", "").replace("+", "")

# Load the appropriate training dataset based on input type
if input_dataset == 'fix':
    price_train = torch.load(f'../Data/GAD_{company_name}/GAD_fix_train.pt', map_location=torch.device('cpu'))
elif input_dataset == 'robust':
    price_train = torch.load(f'../Data/GAD_{company_name}/GAD_robust_train.pt', map_location=torch.device('cpu'))
elif input_dataset == 'real':
    price_train = torch.load(f'../Data/GAD_{company_name}/GAD_real_train.pt', map_location=torch.device('cpu'))
else:
    raise ValueError("input dataset must be 'fix' or 'robust'")

# Calculate batch size and data partitioning
batch_size = max(int(N/20), 250)
sample_size = price_train.shape[0]
print(f'{company_name} sample size: {sample_size}, batch size: {batch_size}, number of runs: {int(sample_size/N)}')

# Loop over data partitions
for part in range(0, int(sample_size/N)):
    index1 = part*N
    index2 = (part+1)*N
    # Prepare partitioned training data and loader
    train_data = torch.utils.data.TensorDataset(price_train[index1:index2])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # Initialize network and loss function for hedging
    network = RNN_BN_simple(1, sequence_length, device).to(device)
    loss_fn = loss_exp_OCE(13,X_max=True, p0_mode='given').to(device)
    # Setup trainable parameters for clean and adversarial losses
    p0_clean = nn.Parameter(torch.tensor(0.10))
    p0_att = nn.Parameter(torch.tensor(0.10))
    # Setup optimizer with model parameters and trainable variables
    opt = torch.optim.Adam([
        {'params': network.parameters()},  # Model parameters
        {'params': [p0_clean, p0_att]}  # Trainable variable with its own learning rate
    ], lr=learning_rate)
    # Learning rate scheduler
    LR_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=alpha_learning_rate)
    # Attacker setup for adversarial training
    attacker = DH_Attacker(loss_exp_OCE(13,X_max=True, p0_mode='calculate').to(device))
    print(f"{name}_part{part} start training")
    
    # Training loop
    for i in range(epoch_num):
        time1 = time.time()
        if i<100 or delta==0.:
            # Train with only clean loss for first 100 epochs or when no attack
            train_result = epoch_att_loader(train_loader, attacker, network, loss_fn, delta, 1., 0., opt)
        else:
            if i == 100:
                # Synchronize p0_att with p0_clean at epoch 100
                with torch.no_grad():
                    p0_att.copy_(p0_clean.detach().clone())
            # Train with both clean and adversarial loss
            train_result = epoch_att_loader(train_loader, attacker, network, loss_fn, delta, alpha1, alpha2, opt)
        time2 = time.time()
        print(f"epoch{i},clean_loss: {train_result[0]:.10f}, att_loss: {train_result[1]:.10f},time: {time2-time1:.1f}s, p0_clean: {p0_clean.item()}, p0_att: {p0_att.item()}")
        # Step the learning rate scheduler
        LR_scheduler.step()

    # Save trained model
    network.to('cpu')
    network.device = 'cpu'
    torch.save(network.state_dict(), f"../Result/{name}_part{int(part)+1}.pt")

