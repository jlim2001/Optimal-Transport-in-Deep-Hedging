import torch
import torch.nn as nn
from Heston_util import *

sequence_length = 30
dt = 1/365
sigma = 2
T = dt * sequence_length
K = 100
s0 = 100
v0 = 0.04
alpha = 1.
b = 0.04
rho = -0.7
alpha_loss = 0.5
transaction_cost_rate = 0.

generator = PathGeneratorHeston(s0=s0, v0=v0, alpha=alpha, b=b, sigma=sigma, rho=rho, timestep=sequence_length, T=T)

data_train = generator.generate(100000)
data_test = generator.generate(1000000)
data_val = generator.generate(100000)

print("Data train shape: ", data_train[0].shape, data_train[1].shape, data_train[2].shape)
torch.save(data_train, '../Data/Heston_train.pt')
print("Data test shape: ", data_test[0].shape, data_test[1].shape, data_test[2].shape)
torch.save(data_test, '../Data/Heston_test.pt')
print("Data val shape: ", data_val[0].shape, data_val[1].shape, data_val[2].shape)
torch.save(data_val, '../Data/Heston_val.pt')

data_OOSP = [torch.Tensor([]), torch.Tensor([]), torch.Tensor([])]
# Initialize the tensor to store the results
parameters_list = []
for i in range(100):
    # Generate out-of-sample paths
    out_of_sample_generator = OutOfSamplePathGeneratorHeston(s0, v0, alpha, b, sigma, rho, sequence_length, T, variation=0.1)
    S_out, V_out, VarPrice_out, params = out_of_sample_generator.generate_out_of_sample_paths(10000)
    parameters_list.append(params)
    data_OOSP[0] = torch.cat((data_OOSP[0], S_out), dim=0)
    data_OOSP[1] = torch.cat((data_OOSP[1], V_out), dim=0)
    data_OOSP[2] = torch.cat((data_OOSP[2], VarPrice_out), dim=0)
    # Loop over the models in the name_list

print("Data OODP shape: ", data_OOSP[0].shape, data_OOSP[1].shape, data_OOSP[2].shape)
torch.save(data_OOSP, '../Data/Heston_OODP.pt')
torch.save(parameters_list, '../Data/Heston_OODP_params.pt')