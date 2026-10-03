import torch
import torch.nn as nn
import argparse
import time
from BS_util import *

# Define parameters for the Black-Scholes model
sequence_length = 30
dt = 1/365
sigma = 0.2
T = dt * sequence_length
K = 100
S0 = 100
# Define the path generator for Black-Scholes model
generator = path_generator_BS(sequence_length, S0, 0, sigma, dt)
# Generate training, testing, and validation datasets
price_train = generator.generate(100000)
price_test = generator.generate(1000000)
price_val = generator.generate(100000)
# Save the generated datasets
print("Price train shape: ", price_train.shape)
torch.save(price_train, '../Data/BS_train.pt')
print("Price test shape: ", price_test.shape)
torch.save(price_test, '../Data/BS_test.pt')
print("Price val shape: ", price_val.shape)
torch.save(price_val, '../Data/BS_val.pt')