import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class RNN_shared(nn.Module):
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
        super(RNN_shared, self).__init__()
        self.input_size = input_size
        self.sequnce_length = sequence_length
        self.device = device
        hidden_size = 256
        self.rnn = nn.ModuleList([nn.Sequential(
                            nn.Linear(self.input_size+1,hidden_size),
                            nn.ReLU(),
                            nn.Linear(hidden_size, hidden_size),
                            nn.ReLU(),
                            nn.Linear(hidden_size, hidden_size),
                            nn.ReLU(),
                            nn.Linear(hidden_size, hidden_size),
                            nn.ReLU(),
                            nn.Linear(hidden_size, self.input_size)) for i in range(sequence_length)])
    
    def forward(self, input):
        input = input.transpose(0, 1)
        time_steps = torch.linspace(0, self.sequnce_length,self.sequnce_length+1, device=self.device).unsqueeze(1).repeat(1, input.shape[1]) / 250
        sequence_length = self.sequnce_length
        assert input.shape[0] == sequence_length, 'input shape is not correct'
        output = torch.Tensor([]).to(self.device)
        for i in range(sequence_length):
            if i == 0:
                input_hidden = torch.cat((input[i], time_steps[i].unsqueeze(-1)), dim=1)
                output_i = self.rnn[i](input_hidden)
            else:
                input_hidden = torch.cat((input[i], time_steps[i].unsqueeze(-1)), dim=1)
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
    
class loss_exp(nn.Module):
    def __init__(self,
                 Strike_price,
                 lamb,
                 X_max=False
                 ):
        super(loss_exp, self).__init__()
        self.K = Strike_price
        self.lamb = lamb
        self.X_max = X_max  
    def terminal_payoff(self, final_price):
        return torch.max(final_price - self.K, torch.zeros_like(final_price))
    
    def exp_utility(self, X):
        return torch.exp(-self.lamb * X).mean().log()/self.lamb

    # def p0_fn(self, X):
    #     d1 = (torch.log(X / self.K) + (0.5 * self.vol ** 2) * self.T) / (self.vol * self.T**0.5)
    #     d2 = d1 - self.vol * self.T**0.5
    #     call_price = X * torch.distributions.Normal(0, 1).cdf(d1) - self.K * torch.distributions.Normal(0, 1).cdf(d2)
    #     return call_price
    
    def forward(self,
                holding, 
                price,
               ):
        delta_price = price[:, 1:] - price[:, :-1]
        PnL = (holding * delta_price).sum(dim=1)
        C_T = self.terminal_payoff(price[:, -1])
        # p_0 = self.p0_fn(price[:, 0])
        X = PnL-C_T
        if self.X_max:
            X = torch.max(X, -torch.ones_like(X) * 10)
        loss = self.exp_utility(X)
        return loss

class loss_exp_OCE(nn.Module):
    def __init__(self,
                 lamb,
                 X_max=False,
                p0_mode='train'
                 ):
        super(loss_exp_OCE, self).__init__()
        self.lamb = lamb
        self.X_max = X_max
        if p0_mode == 'train':
            self.p0 = nn.Parameter(torch.tensor(1.96, requires_grad=True))
        else:
            self.p0 = 1.96
        self.p0_mode = p0_mode
    def terminal_payoff(self, price):
        return torch.max(torch.mean(price,dim=1) - 10, torch.zeros(price.shape[0], device=price.device))
    
    def exp_utility(self, x):
        return torch.exp(-self.lamb * x).mean()+self.p0-(1+np.log(self.lamb))/self.lamb

    def forward(self,
                holding, 
                price,
                p0=None
               ):
        delta_price = price[:, 1:] - price[:, :-1]
        PnL = (holding * delta_price).sum(dim=1)
        C_T = self.terminal_payoff(price)
        X = PnL-C_T
        if self.X_max:
            X = torch.max(X, -torch.ones_like(X))
        if self.p0_mode == 'calculate':
            self.p0 = (torch.exp(-self.lamb * X).mean().log()/self.lamb).item()+np.log(self.lamb)/self.lamb
        elif self.p0_mode == 'train':
            pass
        elif self.p0_mode == 'given':
            self.p0 = p0

        loss = self.exp_utility(X+self.p0)
        return loss

class loss_square(nn.Module):
    def __init__(self,
                p0_mode='train',
                 ):
        super(loss_square, self).__init__()
        # self.K = Strike_price
        if p0_mode == 'train':
            self.p0 = nn.Parameter(torch.tensor(1.96, requires_grad=True))
        else:
            self.p0 = 1.96
        self.p0_mode = p0_mode
    
    def terminal_payoff(self, price):
        return torch.max(torch.mean(price,dim=1) - 10, torch.zeros(price.shape[0], device=price.device))
    
    # def p0_fn(self, X):
    #     d1 = ((X / self.K) + (0.5 * self.vol ** 2) * self.T) / (self.vol * self.T**0.5)
    #     d2 = d1 - self.vol * self.T**0.5
    #     call_price = X * torch.distributions.Normal(0, 1).cdf(d1) - self.K * torch.distributions.Normal(0, 1).cdf(d2)
    #     return call_price
    
    def l_fn(self, X):
        return X.pow(2)
        # return torch.max(X, torch.zeros_like(X)).pow(2)

    def forward(self,
                holding, 
                price,
                p0=None
               ):
        delta_price = price[:, 1:] - price[:, :-1]
        PnL = (holding * delta_price).sum(dim=1)
        C_T = self.terminal_payoff(price)
        X =  PnL - C_T
        if self.p0_mode == 'calculate':
            self.p0 = torch.mean(X)
        if self.p0_mode == 'given':
            if p0 is None:
                raise ValueError("p0 must be provided when p0_mode is 'given'")
            self.p0 = p0
        X = X - self.p0
        loss = self.l_fn(X)
        return loss.mean()
    
class loss_CVAR(nn.Module):
    def __init__(self,
                 alpha_loss,
                 p0_mode = 'search',
                 trans_cost_rate=0.,
                 ):
        super(loss_CVAR, self).__init__()
        self.alpha = alpha_loss
        if p0_mode == 'train':
            self.p0 = nn.Parameter(torch.tensor(1.96, requires_grad=True))
        else:
            self.p0 = 1.96
        self.p0_mode = p0_mode
        self.transaction_cost_rate = trans_cost_rate
    
    def terminal_payoff(self, price):
        return torch.max(torch.mean(price,dim=1) - 10, torch.zeros(price.shape[0], device=price.device))
    
    def forward(self,
                holding, 
                S,
                p0 = None,
                output_hedging_error = False,

               ):
        delta_S = S[:, 1:] - S[:, :-1]
        PnL = (holding * delta_S).sum(dim=1)
        C_T = self.terminal_payoff(S)
        X = C_T - PnL
        if self.transaction_cost_rate > 0:
            transaction_cost = (torch.diff(holding,dim=1,prepend=holding[:,:1]).abs() * self.transaction_cost_rate * S[:,:-1]).sum(dim=(1,2))
            X  = X + transaction_cost
        if self.p0_mode == 'search':
            self.p0 = find_optimal_p0(X,self.alpha)
        elif self.p0_mode == 'given':
            if p0 is None:
                raise ValueError('p0 is not provided')
            self.p0 = p0
        elif self.p0_mode == 'train':
            pass
        X = X - self.p0
        loss = torch.max(X, torch.zeros_like(X))/(1-self.alpha)+self.p0
        if output_hedging_error:
            return loss.mean(), torch.max(X, torch.zeros_like(X)).mean()
        return loss.mean()

def find_optimal_p0(X_total, loss_alpha=0.5, tolerance=1e-4):
    return X_total.quantile(loss_alpha).item()

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
        input_tensor = (price[:,:-1]).unsqueeze(-1)
        holding = network(input_tensor).squeeze()
        PnL = (holding * (price[:, 1:] - price[:, :-1])).sum(dim=1)
        p0 = self.loss_fn.p0_fn(price[:, 0])
        Z = self.loss_fn.terminal_payoff(price[:, -1])
        X_before = Z - p0 - PnL
        alpha = delta*ratio/n
        for i in range(n):
            input_tensor = (price[:,:-1]+att[:,:-1]).unsqueeze(-1)
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
        input_tensor = (price_att[:,:-1]).unsqueeze(-1)
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
            input_tensor = (price_att[:,:-1].unsqueeze(-1)).to(device)
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
        input_tensor = (price_att[:,:-1].unsqueeze(-1)).to(device)
        holding = network(input_tensor).squeeze()
        loss = self.loss_fn(holding, price_att)
        if loss>loss_max:
            loss_max = loss
            att_best = (budget.unsqueeze(1)*att_sign).clone().detach()

        price_att = price+att_best
        input_tensor = (price_att[:,:-1].unsqueeze(-1)).to(device)
        holding = network(input_tensor).squeeze()
        # PnL_att = (holding * (price_att[:, 1:] - price_att[:, :-1])).sum(dim=1)
        # p0_att = self.loss_fn.p0_fn(price_att[:, 0])
        # Z_att = self.loss_fn.terminal_payoff(price_att[:, -1])
        # X_after = Z_att - p0_att - PnL_att
        loss = self.loss_fn(holding, price_att)
        return att_best.cpu(), loss_max
    

class path_generator_GAD(nn.Module):
    def __init__(self,
                 time_steps,
                 S0,
                 a0,
                 a1,
                 b0,
                 b1,
                 gamma,
                 dt):
        super(path_generator_GAD, self).__init__()
        self.time_steps = time_steps+1
        self.init_price = S0
        self.a0, self.a1, self.b0, self.b1, self.gamma = a0, a1, b0, b1, gamma
        self.dt = dt

    def generate(self,sample_size):
        Z = torch.normal(0, np.sqrt(self.dt), (sample_size, self.time_steps))    
        S = torch.zeros((sample_size, self.time_steps))
        S[: ,0] = self.init_price
        for t in range(1, self.time_steps):
            a0_sample = self.a0[0]+(self.a0[1]-self.a0[0])*torch.rand((sample_size,self.time_steps))
            a1_sample = self.a1[0]+(self.a1[1]-self.a1[0])*torch.rand((sample_size,self.time_steps))
            b0_sample = self.b0[0]+(self.b0[1]-self.b0[0])*torch.rand((sample_size,self.time_steps))
            b1_sample = self.b1[0]+(self.b1[1]-self.b1[0])*torch.rand((sample_size,self.time_steps))
            gamma_sample = self.gamma[0]+(self.gamma[1]-self.gamma[0])*torch.rand((sample_size,self.time_steps))
            S[:,t] = S[: ,t - 1] + (b0_sample[:,t]+b1_sample[:,t]*S[: ,t - 1]) * self.dt + \
                    (a0_sample[:,t]+a1_sample[:,t]*torch.max(S[: ,t - 1],torch.zeros_like(S[: ,t - 1]))).pow(gamma_sample[:,t]) * Z[:,t]
        return S