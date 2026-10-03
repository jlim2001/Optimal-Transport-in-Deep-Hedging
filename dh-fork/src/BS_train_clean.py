#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import argparse
import time
from BS_util import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"running on {device}")


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
    total_loss=0.
    for price, in loader:
        price = price.to(device)
        input_tensor = torch.log(price[:,:-1].unsqueeze(-1))
        holding = network(input_tensor).squeeze()
        loss = loss_fn(holding, price, p0=p0_clean)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total_loss += loss.item()
    total_loss /= len(loader)
    return total_loss

# Learning rate schedule function
def alpha_learning_rate(epoch):
    if epoch<100:
        return 1
    elif epoch<200:
        return 0.1
    elif epoch<250:
        return 0.01
    else:
        return 0.001


# Define parameters for the Black-Scholes model
sequence_length = 30
dt = 1/365
learning_rate = 0.005
batch_size = 10000
batch_num=20
epoch_num = 300

sigma = 0.2
T = dt * sequence_length
K = 100
S0 = 100

# Create the parser
parser = argparse.ArgumentParser(description="Script for configuring network parameters.")

# Add arguments with default values
parser.add_argument("--N", type=int, default=10000, help="number of samples.")
args = parser.parse_args()
args_dict = vars(args)
print(args_dict)
N = args.N

# Define the name for saving the model
name = f"BSClean_N{N:.0e}".replace("+0", "").replace("-0", "-")

# Load the training data
price_train = torch.load('../Data/BS_train.pt')
# Loop over data partitions
for part in range(0, int(1e5/N)):
    index1 = part*N
    index2 = (part+1)*N
    # Prepare partitioned training data and loader
    train_data = torch.utils.data.TensorDataset(price_train[index1:index2])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # Initialize network and loss function
    network = RNN_BN_simple(1,sequence_length,device).to(device)
    loss_fn = loss_exp_OCE(K, sigma, T,1.3,X_max=True, p0_mode='given').to(device)
    # Setup Trainable parameters and optimizer
    p0_clean = nn.Parameter(torch.tensor(1.69))
    opt = torch.optim.Adam([
        {'params': network.parameters()},  # Model parameters
        {'params': [p0_clean]}     # Trainable variables
    ], lr=learning_rate)
    # Learning rate scheduler
    LR_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=alpha_learning_rate)

    best_loss = float('inf')
    print(f'{name}_part{part} starts')
    # Training loop
    for i in range(epoch_num):
        time1 = time.time()
        network.train()
        train_result = epoch_loader(train_loader, network, loss_fn, opt)
        time2 = time.time()
        print(f"epoch {i}, train loss: {train_result}, time: {time2-time1}")
        # Step the learning rate scheduler
        LR_scheduler.step()

    # Save trained model
    network.to('cpu')
    network.device = 'cpu'
    torch.save(network.state_dict(), f"../Result/{name}_part{int(part)+1}.pt")
