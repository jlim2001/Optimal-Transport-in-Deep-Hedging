import torch
import torch.nn as nn
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNN_BN(nn.Module):
    """
    A Recurrent Neural Network (RNN) with Batch Normalization (BN) layers.
    This model processes a sequence of inputs and applies a series of linear
    transformations with batch normalization and ReLU activations.

    Args:
        input_size (int): The size of each input vector.
        sequence_length (int): The length of the input sequence.
        device (torch.device): The device to run the model on (e.g., 'cpu' or 'cuda').

    Attributes:
        input_size (int): The size of each input vector.
        sequence_length (int): The length of the input sequence.
        rnn (nn.ModuleList): A list of sequential layers applied to each input in the sequence.
    """
    def __init__(self,
                 input_size,
                 sequence_length,
                device,
                 ):
        super(RNN_BN, self).__init__()
        self.input_size = input_size
        self.sequnce_length = sequence_length
        self.device = device
        self.rnn = nn.ModuleList([nn.Sequential(
                            nn.BatchNorm1d(self.input_size*2),
                            nn.Linear(self.input_size*2, 20),
                            nn.BatchNorm1d(20),
                            nn.ReLU(),
                            nn.Linear(20, 20),
                            nn.BatchNorm1d(20),
                            nn.ReLU(),
                            nn.Linear(20, self.input_size)) for i in range(sequence_length)])
    
    def forward(self, input):
        input = input.transpose(0, 1)
        sequence_length = self.sequnce_length
        assert input.shape[0] == sequence_length, 'input shape is not correct'
        output = torch.Tensor([]).to(self.device)
        output_0 = torch.zeros(input.shape[1], self.input_size).to(self.device)
        for i in range(sequence_length):
            if i == 0:
                input_hidden = torch.cat((input[i], output_0), dim=1)
                output_i = self.rnn[i](input_hidden)
            else:
                input_hidden = torch.cat((input[i], output[i-1]), dim=1)
                output_i = self.rnn[i](input_hidden)
            output = torch.cat((output, output_i.unsqueeze(0)), dim=0)
        return output.transpose(0, 1)

class RNN_BN_simple(nn.Module):
    """
    A Recurrent Neural Network (RNN) with Batch Normalization (BN) layers.
    This model processes a sequence of inputs and applies a series of linear
    transformations with batch normalization and ReLU activations.
    Args:
        input_size (int): The size of each input vector.
        sequence_length (int): The length of the input sequence.
        device (torch.device): The device to run the model on (e.g., 'cpu' or 'cuda').

    Attributes:
        input_size (int): The size of each input vector.
        sequence_length (int): The length of the input sequence.
        rnn (nn.ModuleList): A list of sequential layers applied to each input in the sequence.
    """
    def __init__(self,
                 input_size,
                 sequence_length,
                 device,
                 ):
        super(RNN_BN_simple, self).__init__()
        self.sequnce_length = sequence_length
        self.input_size = input_size
        self.device = device
        self.rnn = nn.ModuleList([nn.Sequential(
                            nn.BatchNorm1d(self.input_size),
                            nn.Linear(self.input_size, 20),
                            nn.BatchNorm1d(20),
                            nn.ReLU(),
                            nn.Linear(20, 20),
                            nn.BatchNorm1d(20),
                            nn.ReLU(),
                            nn.Linear(20, self.input_size)) for i in range(sequence_length)])
        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eps = 1e-3
                module.momentum = 0.3
        
    def forward(self, input):
        input = input.transpose(0, 1)
        sequence_length = self.sequnce_length
        assert input.shape[0] == sequence_length, 'input shape is not correct'
        output = torch.Tensor([]).to(self.device)
        for i in range(sequence_length):
            input_hidden = input[i]
            output_i = self.rnn[i](input_hidden)
            output = torch.cat((output, output_i.unsqueeze(0)), dim=0)
        return output.transpose(0, 1)

class RNN_BN_dropout(nn.Module):
    """
    A Recurrent Neural Network (RNN) with Batch Normalization (BN) layers.
    This model processes a sequence of inputs and applies a series of linear
    transformations with batch normalization and ReLU activations.
    Args:
        input_size (int): The size of each input vector.
        sequence_length (int): The length of the input sequence.
        device (torch.device): The device to run the model on (e.g., 'cpu' or 'cuda').

    Attributes:
        input_size (int): The size of each input vector.
        sequence_length (int): The length of the input sequence.
        rnn (nn.ModuleList): A list of sequential layers applied to each input in the sequence.
    """
    def __init__(self,
                 input_size,
                 sequence_length,
                 device,
                 ):
        super(RNN_BN_dropout, self).__init__()
        self.sequnce_length = sequence_length
        self.input_size = input_size
        self.device = device
        self.rnn = nn.ModuleList([nn.Sequential(
                            nn.BatchNorm1d(self.input_size),
                            nn.Linear(self.input_size, 20),
                            nn.BatchNorm1d(20),
                            nn.ReLU(),
                            nn.Dropout(0.05),
                            nn.Linear(20, 20),
                            nn.BatchNorm1d(20),
                            nn.ReLU(),
                            nn.Dropout(0.05),
                            nn.Linear(20, self.input_size)) for i in range(sequence_length)])
        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eps = 1e-3
                module.momentum = 0.3
        
    def forward(self, input):
        input = input.transpose(0, 1)
        sequence_length = self.sequnce_length
        assert input.shape[0] == sequence_length, 'input shape is not correct'
        output = torch.Tensor([]).to(self.device)
        for i in range(sequence_length):
            input_hidden = input[i]
            output_i = self.rnn[i](input_hidden)
            output = torch.cat((output, output_i.unsqueeze(0)), dim=0)
        return output.transpose(0, 1)
    
class RNN_simple(nn.Module):
    """
    A Recurrent Neural Network (RNN) with Batch Normalization (BN) layers.
    This model processes a sequence of inputs and applies a series of linear
    transformations with batch normalization and ReLU activations.
    Args:
        input_size (int): The size of each input vector.
        sequence_length (int): The length of the input sequence.
        device (torch.device): The device to run the model on (e.g., 'cpu' or 'cuda').

    Attributes:
        input_size (int): The size of each input vector.
        sequence_length (int): The length of the input sequence.
        rnn (nn.ModuleList): A list of sequential layers applied to each input in the sequence.
    """
    def __init__(self,
                 input_size,
                 sequence_length,
                 device,
                 ):
        super(RNN_simple, self).__init__()
        self.sequnce_length = sequence_length
        self.input_size = input_size
        self.device = device
        self.rnn = nn.ModuleList([nn.Sequential(
                            nn.Linear(self.input_size, 20),
                            nn.ReLU(),
                            nn.Linear(20, 20),
                            nn.ReLU(),
                            nn.Linear(20, self.input_size)) for i in range(sequence_length)])
    
    def forward(self, input):
        input = input.transpose(0, 1)
        sequence_length = self.sequnce_length
        assert input.shape[0] == sequence_length, 'input shape is not correct'
        output = torch.Tensor([]).to(self.device)
        for i in range(sequence_length):
            input_hidden = input[i]
            output_i = self.rnn[i](input_hidden)
            output = torch.cat((output, output_i.unsqueeze(0)), dim=0)
        return output.transpose(0, 1)

class loss_exp(nn.Module):
    def __init__(self,
                 Strike_price,
                    sigma,
                    T,
                 lamb,
                 X_max=False
                 ):
        super(loss_exp, self).__init__()
        self.K = Strike_price
        self.vol = sigma
        self.T = T
        self.lamb = lamb
        self.X_max = X_max  
    def terminal_payoff(self, final_price):
        return torch.max(final_price - self.K, torch.zeros_like(final_price))
    
    def exp_utility(self, X):
        return torch.exp(-self.lamb * X).mean().log()/self.lamb

    def p0_fn(self, X):
        d1 = (torch.log(X / self.K) + (0.5 * self.vol ** 2) * self.T) / (self.vol * self.T**0.5)
        d2 = d1 - self.vol * self.T**0.5
        call_price = X * torch.distributions.Normal(0, 1).cdf(d1) - self.K * torch.distributions.Normal(0, 1).cdf(d2)
        return call_price
    
    def forward(self,
                holding, 
                price,
               ):
        delta_price = price[:, 1:] - price[:, :-1]
        PnL = (holding * delta_price).sum(dim=1)
        C_T = self.terminal_payoff(price[:, -1])
        p_0 = self.p0_fn(price[:, 0])
        X = PnL-C_T
        if self.X_max:
            X = torch.max(X, -torch.ones_like(X) * 10)
        loss = self.exp_utility(X)
        return loss

class loss_exp_OCE(nn.Module):
    def __init__(self,
                 Strike_price,
                    sigma,
                    T,
                 lamb,
                 X_max=False,
                p0_mode='train'
                 ):
        super(loss_exp_OCE, self).__init__()
        self.K = Strike_price
        self.vol = sigma
        self.T = T
        self.lamb = lamb
        self.X_max = X_max
        if p0_mode == 'train':
            self.p0 = nn.Parameter(torch.tensor(1.96, requires_grad=True))
        else:
            self.p0 = 1.96
        self.p0_mode = p0_mode
    def terminal_payoff(self, final_price):
        return torch.max(final_price - self.K, torch.zeros_like(final_price))
    
    def exp_utility(self, x):
        return torch.exp(-self.lamb * x).mean()+self.p0-(1+np.log(self.lamb))/self.lamb

    def p0_fn(self, X):
        d1 = (torch.log(X / self.K) + (0.5 * self.vol ** 2) * self.T) / (self.vol * self.T**0.5)
        d2 = d1 - self.vol * self.T**0.5
        call_price = X * torch.distributions.Normal(0, 1).cdf(d1) - self.K * torch.distributions.Normal(0, 1).cdf(d2)
        return call_price
    
    def forward(self,
                holding, 
                price,
                mode='test'
               ):
        delta_price = price[:, 1:] - price[:, :-1]
        PnL = (holding * delta_price).sum(dim=1)
        C_T = self.terminal_payoff(price[:, -1])
        X = PnL-C_T
        if self.X_max:
            X = torch.max(X, -torch.ones_like(X) * 10)
        if self.p0_mode == 'calculate':
            self.p0 = (torch.exp(-self.lamb * X).mean().log()/self.lamb).item()+np.log(self.lamb)/self.lamb
        elif self.p0_mode == 'train':
            pass

        loss = self.exp_utility(X+self.p0)
        return loss


class loss_square(nn.Module):
    def __init__(self,
                 Strike_price,
                 vol,
                 T,
                 ):
        super(loss_square, self).__init__()
        self.K = Strike_price
        self.vol = vol
        self.T = T
    
    def terminal_payoff(self, final_price):
        return torch.max(final_price - self.K, torch.zeros_like(final_price))
    
    def p0_fn(self, X):
        d1 = (torch.log(X / self.K) + (0.5 * self.vol ** 2) * self.T) / (self.vol * self.T**0.5)
        d2 = d1 - self.vol * self.T**0.5
        call_price = X * torch.distributions.Normal(0, 1).cdf(d1) - self.K * torch.distributions.Normal(0, 1).cdf(d2)
        return call_price
    
    def l_fn(self, X):
        return X.pow(2)
        # return torch.max(X, torch.zeros_like(X)).pow(2)

    def forward(self,
                holding, 
                price,
               ):
        delta_price = price[:, 1:] - price[:, :-1]
        PnL = (holding * delta_price).sum(dim=1)
        C_T = self.terminal_payoff(price[:, -1])
        p_0 = self.p0_fn(price[:, 0])
        X = C_T - p_0 - PnL
        loss = self.l_fn(X)
        return loss.mean()

class path_generator_BS(nn.Module):
    def __init__(self,
                 time_steps,
                 S0,
                 mean,
                 vol,
                 dt):
        super(path_generator_BS, self).__init__()
        self.time_steps = time_steps+1
        self.init_price = S0
        self.mean = mean
        self.vol = vol
        self.dt = dt

    def generate(self,sample_size):
        drift = self.mean - 0.5 * self.vol ** 2
        Z = torch.normal(0, np.sqrt(self.dt), (sample_size, self.time_steps))    
        S = torch.zeros((sample_size, self.time_steps))
        S[: ,0] = self.init_price
        for t in range(1, self.time_steps):
            S[:,t] = S[: ,t - 1] * torch.exp(drift * self.dt + self.vol * Z[:,t])
        return S
        
class path_generator_BS_OOSP(nn.Module):
    def __init__(self,
                 time_steps,
                 S0,
                 mean,
                 vol,
                 dt):
        super(path_generator_BS_OOSP, self).__init__()
        self.time_steps = time_steps+1
        self.init_price = S0
        self.mean = mean
        self.vol = vol
        self.dt = dt

    def generate_instances(self, sigma, n_per_instance):
        n_instance = sigma.shape[0]
        drift = self.mean - 0.5 * sigma ** 2 # n_instance
        Z = torch.normal(0, np.sqrt(self.dt), (n_instance,n_per_instance, self.time_steps))    
        S = torch.zeros((n_instance,n_per_instance, self.time_steps))
        S[:, : ,0] = self.init_price
        for t in range(1, self.time_steps):
            S[:,:,t] = S[:,:,t - 1] * torch.exp(drift.unsqueeze(-1) * self.dt + sigma.unsqueeze(-1) * Z[:,:,t])
        return S
    
    def generate_OOSP(self, n_instance=10, n_per_instance=10000):
        Z = torch.normal(0, 1, (n_instance, 300))
        sigma_sample = self.vol / torch.std(Z,dim=1,correction=1)
        S = self.generate_instances(sigma_sample, n_per_instance)
        return S, sigma_sample
    
    def sigma_sample(self, n_instance=10):
        Z = torch.normal(0, 1, (n_instance, self.time_steps))
        sigma_sample = self.vol / torch.std(Z,dim=1,correction=1)
        return sigma_sample

    def generate(self, n_instance=100, n_per_instance=10000):
        S = self.generate_OOSP(n_instance, n_per_instance)[0]
        S = S.view(-1, self.time_steps)
        return S

class model_hedge():
    def __init__(self,
                 S0,
                 K,
                 mean,
                 vol,
                 dt,
                 sequence_length,
                 T,
                 ):
        self.S0 = S0
        self.K = K
        self.mean = mean
        self.vol = vol
        self.dt = dt
        self.sequence_length = sequence_length
        self.T = T
    
    def strategy(self, X):
        time_to_maturity = (self.T - self.dt * torch.arange(0, self.sequence_length)).unsqueeze(0).to(device)
        d1 = (torch.log(X / self.K) + (0.5 * self.vol ** 2) * time_to_maturity) / (self.vol * time_to_maturity**0.5)
        try:
            holding = torch.distributions.Normal(0, 1).cdf(d1)
        except ValueError as e:
            # Catch and print the error message
            print(f"ValueError occurred: {e}")

            # Debug and print the problematic tensor
            print("Inspecting the tensor:")
            print("Tensor with NaN values:", d1[torch.isnan(d1)])
            print("Tensor with Inf values:", d1[torch.isinf(d1)])
            print("Tensor min:", d1.min().item())
            print("Tensor max:", d1.max().item())
        return holding
        
class DH_Attacker():
    def __init__(self,
                 loss_fn,
                 ):
        self.loss_fn = loss_fn
        
    def Wp_att(self, network, price, delta, ratio, n, q=2.):
        if delta == 0:
            return price, 0, 0
        att = torch.zeros_like(price)
        att.requires_grad = True
        input_tensor = torch.log(price[:,:-1]).unsqueeze(-1)
        holding = network(input_tensor).squeeze()
        PnL = (holding * (price[:, 1:] - price[:, :-1])).sum(dim=1)
        p0 = self.loss_fn.p0_fn(price[:, 0])
        Z = self.loss_fn.terminal_payoff(price[:, -1])
        X_before = Z - p0 - PnL
        alpha = delta*ratio/n
        for i in range(n):
            input_tensor = torch.log(price[:,:-1]+att[:,:-1]).unsqueeze(-1)
            holding = network(input_tensor).squeeze()
            loss = self.loss_fn(holding, price+att)
            grad = torch.autograd.grad(loss, att)[0]
            grad_norm = grad.norm(p=1,dim=1, keepdim=True)
            #att
            att = att + alpha*torch.sign(grad)*grad_norm.pow(q-1)*((grad_norm.pow(q).mean()+1e-10).pow(1/q-1))
            #proj
            dist = att.norm(p=float('inf'), dim=1, keepdim=True)
            p = (1-1/q)**(-1)
            r = min(1.0, delta/(dist.pow(p).mean().pow(1/q)+1e-10))
            att = torch.clamp(att, -r*dist, r*dist)
            att[:, 0] = 0
            att = torch.clamp(att, -price+0.01)
        price_att = price+att
        input_tensor = torch.log(price_att[:,:-1]).unsqueeze(-1)
        holding = network(input_tensor).squeeze()
        PnL_att = (holding * (price_att[:, 1:] - price_att[:, :-1])).sum(dim=1)
        p0_att = self.loss_fn.p0_fn(price_att[:, 0])
        Z_att = self.loss_fn.terminal_payoff(price_att[:, -1])
        X_after = Z_att - p0_att - PnL_att
        return price_att.detach().clone(), X_before, X_after
    
    def budget_att(self, network, price, delta, ratio, n, q=2.0):
        if delta == 0:
            return torch.zeros_like(price), 0, 0
        #define parameters
        budget = torch.ones(price.shape[0]).to(device)*delta
        budget.requires_grad = True
        att_sign = torch.ones_like(price).sign().to(device)
        att_sign_old = att_sign.clone().detach()
        att_sign[:,0] = 0
        att_sign.requires_grad = True
        #best statistics
        att_best = (budget.unsqueeze(1)*att_sign).clone().detach()
        loss_max = 0.
        alpha = delta*ratio/n
        for i in range(n):
            price_att = price+budget.unsqueeze(1)*att_sign
            input_tensor = torch.log(price_att[:,:-1].unsqueeze(-1)).to(device)
            holding = network(input_tensor).squeeze()
            loss = self.loss_fn(holding, price_att)
            if loss>loss_max:
                loss_max = loss
                att_best = (budget.unsqueeze(1)*att_sign).clone().detach()
            grad_b = torch.autograd.grad(loss, budget,retain_graph=True)[0]
            grad_a = torch.autograd.grad(loss, att_sign,retain_graph=True)[0]
            with torch.no_grad():
                budget_new = budget + alpha * grad_b.pow(q - 1) * ((grad_b.pow(q).mean() + 1e-10).pow(1 / q - 1))
                budget_new = budget_new / budget_new.square().mean().sqrt() * delta
                budget_new = torch.max(budget_new, torch.zeros_like(budget_new))
                budget.copy_(budget_new)
                att_sign_new = (att_sign+0.4*0.75*grad_a.sign()+0.25*(att_sign-att_sign_old)).clamp(-1,1)
                att_sign_old = att_sign
                att_sign_new[:,0] = 0
                att_sign.copy_(att_sign_new)
        price_att = price+budget.unsqueeze(1)*att_sign
        input_tensor = torch.log(price_att[:,:-1].unsqueeze(-1)).to(device)
        holding = network(input_tensor).squeeze()
        loss = self.loss_fn(holding, price_att)
        if loss>loss_max:
            loss_max = loss
            att_best = (budget.unsqueeze(1)*att_sign).clone().detach()

        price_att = price+att_best
        input_tensor = torch.log(price_att[:,:-1].unsqueeze(-1)).to(device)
        holding = network(input_tensor).squeeze()
        PnL_att = (holding * (price_att[:, 1:] - price_att[:, :-1])).sum(dim=1)
        p0_att = self.loss_fn.p0_fn(price_att[:, 0])
        Z_att = self.loss_fn.terminal_payoff(price_att[:, -1])
        X_after = Z_att - p0_att - PnL_att
        loss = self.loss_fn(holding, price_att)
        return att_best.cpu(), loss_max, X_after


def net_budget_att(network,loss_fn, price, delta, ratio, n, q=2.0):
    network.eval()
    att = delta * torch.ones_like(price).sign().to(device)
    att[:,0] = 0
    att.requires_grad = True
    att_sign_old = att.sign()
    att_best = torch.zeros_like(price)
    loss_best = 0
    input_tensor = torch.log(price[:,:-1].unsqueeze(-1)).to(device)
    holding = network(input_tensor).squeeze()
    PnL = (holding * (price[:, 1:] - price[:, :-1])).sum(dim=1)
    p0 = loss_fn.p0_fn(price[:, 0])
    Z = loss_fn.terminal_payoff(price[:, -1])
    X_before = Z - p0 - PnL
    alpha = delta*ratio/n
    for i in range(n):
        input_tensor = torch.log(price[:,:-1].unsqueeze(-1)+att[:,:-1].unsqueeze(-1)).to(device)
        holding = network(input_tensor).squeeze()
        loss = loss_fn(holding, price+att)
        if loss>loss_best:
            loss_best = loss
            att_best = att.clone().detach()
        grad = torch.autograd.grad(loss, att)[0]
        with torch.no_grad():
            budget = att.abs().max(dim=1)[0]
            att_sign_now = (att / budget.unsqueeze(1)).nan_to_num(0)
            grad_b = (grad*att.sign()).sum(dim=1)
            budget_new = budget + alpha * grad_b.pow(q - 1) * ((grad_b.pow(q).mean() + 1e-10).pow(1 / q - 1))
            budget_new = budget_new / budget_new.square().mean().sqrt() * delta
            budget_new = torch.max(budget_new, torch.zeros_like(budget_new))
            att_sign_new = (att_sign_now+0.4*0.75*grad.sign()+0.25*(att_sign_now-att_sign_old)).clamp(-1,1)
            att_sign_old = att_sign_now
            att_new = budget_new.unsqueeze(1) * att_sign_new
            att_new[:,0] = 0
            att.copy_(att_new)
    input_tensor = torch.log(price[:,:-1].unsqueeze(-1)+att[:,:-1].unsqueeze(-1)).to(device)
    holding = network(input_tensor).squeeze()
    loss = loss_fn(holding, price+att)
    if loss>loss_best:
        loss_best = loss
        att_best = att.clone().detach()

    price_att = price+att_best
    input_tensor = torch.log(price_att[:,:-1].unsqueeze(-1)).to(device)
    holding = network(input_tensor).squeeze()
    PnL_att = (holding * (price_att[:, 1:] - price_att[:, :-1])).sum(dim=1)
    p0_att = loss_fn.p0_fn(price_att[:, 0])
    Z_att = loss_fn.terminal_payoff(price_att[:, -1])
    X_after = Z_att - p0_att - PnL_att
    loss = loss_fn(holding, price_att)
    return att_best.cpu(), loss_best, X_after