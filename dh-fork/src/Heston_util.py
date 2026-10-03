import torch
import torch.nn as nn
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PathGeneratorHeston(nn.Module):
    def __init__(self, s0, v0, alpha, b, sigma, rho, timestep, T):
        """
        Initializes the Heston path generator.
        Parameters:
        - s0: Initial stock price.
        - v0: Initial variance.
        - alpha: Mean-reversion rate of variance.
        - b: Long-run variance level.
        - sigma: Volatility of variance.
        - rho: Correlation between asset and variance shocks.
        - timestep: Number of time steps.
        - T: Total simulation time.
        """
        super(PathGeneratorHeston, self).__init__()
        self.s0 = s0
        self.v0 = v0
        self.alpha = alpha
        self.b = b
        self.sigma = sigma
        self.rho = rho
        self.timestep = timestep
        self.dt = T / timestep  # Time step size
        self.T = T

    def logSstep(self, vprev, vnext, dt, sample_size):
        """
        Auxiliary update function for simulating the stock price using the Broadie-Kaya scheme.

        Parameters:
        - vprev: Variance at the previous time step.
        - vnext: Variance at the next time step.
        - dt: Time step size.

        Returns:
        - Logarithmic increment for the stock price.
        """
        vintapprox = 0.5 * (vprev + vnext) * dt  # Approximation of the integrated variance
        normal_increment = np.random.normal(0, np.sqrt(vintapprox), sample_size)  # Random normal term
        # Corrected Term 1
        term1 = (self.rho / self.sigma * self.alpha - 0.5) * vintapprox
        term2 = (self.rho / self.sigma) * (vnext - vprev - self.alpha * self.b * dt)
        term3 = np.sqrt(1 - self.rho**2) * normal_increment
        return term1 + term2 + term3

    def generate_paths(self, sample_size):
        """
        Generates Heston paths for stock price and variance.

        Parameters:
        - sample_size: Number of paths to simulate.

        Returns:
        - S: Simulated stock prices (sample_size x (timestep + 1)).
        - V: Simulated variances (sample_size x (timestep + 1)).
        """

        # Initialize paths
        S = np.zeros((sample_size, self.timestep + 1))
        V = np.zeros((sample_size, self.timestep + 1))

        S[:, 0] = self.s0
        V[:, 0] = self.v0

        d = 4 * self.b * self.alpha / self.sigma**2
        c = self.sigma**2 * (1 - np.exp(-self.alpha * self.dt)) / (4 * self.alpha)

        for t in range(self.timestep):
            vprev = V[:, t]
            lamb = vprev * np.exp(-self.alpha * self.dt) / c
            poisson_term = np.random.poisson(lamb / 2, sample_size)
            V[:, t + 1] = c * np.random.gamma((d + 2 * poisson_term) / 2, 2, sample_size)

            S[:, t + 1] = S[:, t] * np.exp(
                    self.logSstep(V[:, t], V[:, t + 1], self.dt, sample_size)
                )

        return S, V

    def compute_var_swap_prices(self, V):
        """
        Computes the variance swap prices for the given variance paths.

        Parameters:
        - V: Simulated variance paths (sample_size x (timestep + 1)).

        Returns:
        - VarPrice: Variance swap prices (sample_size x (timestep + 1)).
        """
        sample_size = V.shape[0]
        VarPrice = np.zeros((sample_size, self.timestep + 1))

        varInt = torch.zeros(sample_size,)  # Cumulative variance integral
        VarPrice[:, 0] = (V[:, 0] - self.b) / self.alpha * (1 - np.exp(-self.alpha * self.T )) + self.b * self.T
        for i in range(self.timestep):
            # Trapezoidal integration of variance
            varInt += 0.5 * self.dt * (V[:, i] + V[:, i + 1])

            # Correction for mean-reversion
            correction = (V[:, i + 1] - self.b) / self.alpha * (1 - np.exp(-self.alpha * (self.T - (i + 1) * self.dt))) + self.b * (self.T - (i + 1) * self.dt)

            # Add to variance swap price
            VarPrice[:, i + 1] = varInt + correction

        return VarPrice

    def generate(self, sample_size):
        """
        Generates stock prices, variances, and variance swap prices.

        Parameters:
        - sample_size: Number of paths to simulate.

        Returns:
        - S: Simulated stock prices (sample_size x (timestep + 1)).
        - V: Simulated variances (sample_size x (timestep + 1)).
        - VarPrice: Variance swap prices (sample_size x (timestep + 1)).
        """
        S, V = self.generate_paths(sample_size)
        VarPrice = self.compute_var_swap_prices(V)
        return torch.Tensor(S), torch.Tensor(V), torch.Tensor(VarPrice)
    
class OutOfSamplePathGeneratorHeston(PathGeneratorHeston):
    def __init__(self, s0, v0, alpha, b, sigma, rho, timestep, T, variation=0.1,one_side=False):
        """
        Initializes the Heston path generator with parameter variation for out-of-sample data.
        Parameters:
        - s0: Initial stock price.
        - v0: Initial variance.
        - alpha: Mean-reversion rate of variance.
        - b: Long-run variance level.
        - sigma: Volatility of variance.
        - rho: Correlation between asset and variance shocks.
        - timestep: Number of time steps.
        - T: Total simulation time.
        - variation: Percentage variation for parameters.
        """
        super(OutOfSamplePathGeneratorHeston, self).__init__(s0, v0, alpha, b, sigma, rho, timestep, T)
        self.variation = variation
        self.one_side = one_side

    def generate_random_parameters(self):
        """
        Generates random parameters within the specified variation range.
        Ensures that 2 * alpha * b > sigma^2.
        """
        if self.one_side:
            alpha = self.alpha * (1 + np.random.uniform(0, self.variation))
            b = self.b * (1 + np.random.uniform(0, self.variation))
            sigma = self.sigma * (1 + np.random.uniform(0, self.variation))
            rho = self.rho * (1 + np.random.uniform(0, self.variation))
        else:
            alpha = self.alpha * (1 + np.random.uniform(-self.variation, self.variation))
            b = self.b * (1 + np.random.uniform(-self.variation, self.variation))
            sigma = self.sigma * (1 + np.random.uniform(-self.variation, self.variation))
            rho = self.rho * (1 + np.random.uniform(-self.variation, self.variation))
        return alpha, b, sigma, rho

    def generate_out_of_sample_paths(self, sample_size):
        """
        Generates out-of-sample Heston paths for stock price and variance with varied parameters.

        Parameters:
        - sample_size: Number of paths to simulate.

        Returns:
        - S: Simulated stock prices (sample_size x (timestep + 1)).
        - V: Simulated variances (sample_size x (timestep + 1)).
        """
        alpha, b, sigma, rho = self.generate_random_parameters()
        self.alpha, self.b, self.sigma, self.rho = alpha, b, sigma, rho
        S,V =  self.generate_paths(sample_size)
        VarPrice = self.compute_var_swap_prices(V)
        return torch.Tensor(S), torch.Tensor(V), torch.Tensor(VarPrice), (alpha, b, sigma, rho)
    
class RNN(nn.Module):
    def __init__(self,
                 sequence_length,
                 ):
        super(RNN, self).__init__()
        self.sequnce_length = sequence_length
        self.rnn = nn.ModuleList([nn.Sequential(nn.Linear(2, 20), 
                             nn.ReLU(), 
                             nn.Linear(20, 1),
                             nn.Tanh()) for i in range(sequence_length)])
    
    def forward(self, input):
        input = input.transpose(0, 1) # change batch dim and time dim
        sequence_length = self.sequnce_length
        assert input.shape[0] == sequence_length, 'input shape is not correct'
        output = torch.Tensor([]).to(device)
        output_0 = torch.zeros(input.shape[1], 1).to(device)
        for i in range(sequence_length):
            if i == 0:
                input_hidden = torch.cat((input[i], output_0), dim=1)
                output_i = self.rnn[i](input_hidden)
            else:
                input_hidden = torch.cat((input[i], output[i-1]), dim=1)
                output_i = self.rnn[i](input_hidden)
            output = torch.cat((output, output_i.unsqueeze(0)), dim=0)
        return output.transpose(0, 1)   

class RNN_simple(nn.Module):
    def __init__(self,
                 sequence_length,
                 ):
        super(RNN_simple, self).__init__()
        self.sequnce_length = sequence_length
        self.rnn = nn.ModuleList([nn.Sequential(nn.Linear(1, 20), 
                             nn.ReLU(), 
                             nn.Linear(20, 1),
                             nn.Tanh()) for i in range(sequence_length)])
    
    def forward(self, input):
        input = input.transpose(0, 1)
        sequence_length = self.sequnce_length
        assert input.shape[0] == sequence_length, 'input shape is not correct'
        output = torch.Tensor([]).to(device)
        for i in range(sequence_length):
            input_hidden = input[i]
            output_i = self.rnn[i](input_hidden)
            output = torch.cat((output, output_i.unsqueeze(0)), dim=0)
        return output.transpose(0, 1)

class RNN_BN(nn.Module):
    def __init__(self,
                 sequence_length,
                 ):
        super(RNN_BN, self).__init__()
        self.sequnce_length = sequence_length
        self.rnn = nn.ModuleList([nn.Sequential(
                            nn.BatchNorm1d(4,momentum=0.01),
                            nn.Linear(4, 20),
                            nn.BatchNorm1d(20,momentum=0.01),
                            nn.ReLU(),
                            nn.Linear(20, 20),
                            nn.BatchNorm1d(20,momentum=0.01),
                            nn.ReLU(),
                            nn.Linear(20, 2)) for i in range(sequence_length)])
    
    def forward(self, input):
        input = input.transpose(0, 1)
        sequence_length = self.sequnce_length
        assert input.shape[0] == sequence_length, 'input shape is not correct'
        output = torch.Tensor([]).to(device)
        output_0 = torch.zeros(input.shape[1], 2).to(device)
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
    def __init__(self,
                 sequence_length,
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                 ):
        super(RNN_BN_simple, self).__init__()
        self.sequnce_length = sequence_length
        self.rnn = nn.ModuleList([nn.Sequential(
                            nn.BatchNorm1d(2,momentum=0.001),
                            nn.Linear(2, 20),
                            nn.BatchNorm1d(20,momentum=0.001),
                            nn.ReLU(),
                            nn.Linear(20, 20),
                            nn.BatchNorm1d(20,momentum=0.001),
                            nn.ReLU(),
                            nn.Linear(20, 2)) for i in range(sequence_length)])
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eps = 1e-3
                module.momentum = 0.3
        self.device = device
    
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
        time_to_maturity = (self.T - self.dt * torch.arange(0, self.sequence_length)).unsqueeze(0)
        d1 = (torch.log(X / self.K) + (0.5 * self.vol ** 2) * time_to_maturity) / (self.vol * time_to_maturity**0.5)
        holding = torch.distributions.Normal(0, 1).cdf(d1)
        return holding
        
class loss_exp(nn.Module):
    def __init__(self,
                 Strike_price,
                 utility_fn
                 ):
        super(loss_exp, self).__init__()
        self.K = Strike_price

    def terminal_payoff(self, final_price):
        return torch.max(final_price - self.K, torch.zeros_like(final_price))
    
    def exp_utility(self, X):
        return torch.exp(-0.001 * X)

    def forward(self,
                holding, 
                price,
               ):
        delta_price = price[:, 1:] - price[:, :-1]
        PnL = (holding * delta_price).sum(dim=1)
        C_T = self.terminal_payoff(price[:, -1])
        X = PnL - C_T
        loss = self.exp_utility(X)
 
        return loss.mean()
    
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

class loss_p0trained(nn.Module):
    def __init__(self,
                 Strike_price,
                 vol,
                 T,
                 ):
        super(loss_p0trained, self).__init__()
        self.K = Strike_price
        self.vol = vol
        self.T = T
        self.p_0 = nn.Parameter(torch.tensor(1.0))
        nn.init.constant_(self.p_0, self.p0_fn(torch.tensor(100.0)))
    
    def terminal_payoff(self, final_price):
        return torch.max(final_price - self.K, torch.zeros_like(final_price))
    
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
        p_0 = self.p_0
        loss = (C_T - p_0 - PnL).pow(2)
        return loss.mean()
    
class loss_CVAR(nn.Module):
    def __init__(self,
                 Strike_price,
                 vol,
                 T,
                 alpha_loss,
                 p0_mode = 'search',
                 trans_cost_rate=0.,
                 ):
        super(loss_CVAR, self).__init__()
        self.K = Strike_price
        self.vol = vol
        self.T = T
        self.alpha = alpha_loss
        if p0_mode == 'train':
            self.p0 = nn.Parameter(torch.tensor(1.96, requires_grad=True))
        else:
            self.p0 = 1.96
        self.p0_mode = p0_mode
        self.transaction_cost_rate = trans_cost_rate
    
    def terminal_payoff(self, final_price):
        return torch.max(final_price - self.K, torch.zeros_like(final_price))

    def forward(self,
                holding, 
                S,
                VarPrice,
                output_hedging_error = False,
                p0 = None
               ):
        delta_S = S[:, 1:] - S[:, :-1]
        delta_V = VarPrice[:, 1:] - VarPrice[:, :-1]
        delta_price = torch.cat((delta_S.unsqueeze(-1), delta_V.unsqueeze(-1)), dim=2)
        PnL = (holding * delta_price).sum(dim=(1,2))
        C_T = self.terminal_payoff(S[:,-1])
        X = C_T - PnL
        if self.transaction_cost_rate > 0:
            price = torch.cat((S.unsqueeze(-1), VarPrice.unsqueeze(-1)), dim=2)
            transaction_cost = (torch.diff(holding,dim=1,prepend=holding[:,:1]).abs() * self.transaction_cost_rate * price[:,:-1,:]).sum(dim=(1,2))
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

class Heston_Attacker():
    def __init__(self,
                 loss_fn,
                    s0,
                    v0,
                    alpha,
                    b,
                    sigma,
                    rho,
                    timestep,
                    T,
                 ):
        self.loss_fn = loss_fn
        self.s0 = s0
        self.v0 = v0
        self.alpha = alpha
        self.b = b
        self.sigma = sigma
        self.rho = rho
        self.timestep = timestep
        self.dt = T / timestep  # Time step size
        self.T = T

    def V_to_VarPrice(self,V):
        """
        Computes the variance swap prices for the given variance paths.
        Parameters:
        - V: Simulated variance paths (sample_size x (timestep + 1)).
        Returns:
        - VarPrice: Variance swap prices (sample_size x (timestep + 1)).
        """
        sequence_length = V.shape[1] - 1
        sample_size = V.shape[0]
        VarPrice = torch.zeros((sample_size, sequence_length + 1), device=V.device)
        varInt = torch.zeros(sample_size, device=V.device)  # Cumulative variance integral
        VarPrice[:, 0] = (V[:, 0] - self.b) / self.alpha * (1 - np.exp(-self.alpha * self.T)) + self.b * self.T

        # Trapezoidal integration of variance
        varInt = 0.5 * self.dt * (V[:, :-1] + V[:, 1:]).cumsum(dim=1)

        # Correction for mean-reversion
        time_factors = torch.arange(1, sequence_length + 1, device=V.device) * self.dt
        correction = (V[:, 1:] - self.b) / self.alpha * (1 - torch.exp(-self.alpha * (self.T - time_factors))) + self.b * (self.T - time_factors)

        # Add to variance swap price
        VarPrice[:, 1:] = varInt + correction
        return VarPrice 
       
    def net_W2_att_SV(self,network, S,V, delta, ratio, n,q=2,):
        sequence_length = S.shape[1] - 1
        att = torch.zeros(S.shape[0], sequence_length+1, 2).to(device)
        att_min = torch.cat((-S.unsqueeze(-1), -V.unsqueeze(-1)), dim=2)
        att.requires_grad = True
        alpha = delta*ratio/n
        for i in range(n):
            S_att = S + att[:, :, 0]
            V_att = V + att[:, :, 1]/100
            VarPrice_att = self.V_to_VarPrice(V_att)
            input_vector = torch.cat((torch.log(S_att[:, :-1]).unsqueeze(-1), V_att[:, :-1].unsqueeze(-1)), dim=-1)
            holding = network(input_vector).squeeze()
            loss = self.loss_fn(holding, S_att, VarPrice_att)
            
            grad = torch.autograd.grad(loss, att)[0]*S.shape[0]
            grad[:,0,:] = 0
            grad_norm = grad.norm(p=1,dim=1, keepdim=True)
            #att
            att = att + alpha*torch.sign(grad)*grad_norm.pow(q-1)*((grad_norm.pow(q).mean()+1e-10).pow(1/q-1))
            #proj
            dist = att.norm(p=float('inf'), dim=1, keepdim=True)
            p = (1-1/q)**(-1)
            r = min(1.0, delta/(dist.pow(p).mean().pow(1/q)+1e-10))
            att = torch.clamp(att, -r*dist, r*dist)
            att[:,0,:] = 0
            att = torch.clamp(att,att_min)

        S_att = S + att[:, :, 0]
        V_att = V + att[:, :, 1]/100
        VarPrice_att = self.V_to_VarPrice(V_att)
        input_vector = torch.cat((torch.log(S_att[:, :-1]).unsqueeze(-1), V_att[:, :-1].unsqueeze(-1)), dim=-1)
        holding = network(input_vector).squeeze()
        delta_S = S_att[:, 1:] - S_att[:, :-1]
        delta_V = VarPrice_att[:, 1:] - VarPrice_att[:, :-1]
        delta_price = torch.cat((delta_S.unsqueeze(-1), delta_V.unsqueeze(-1)), dim=2)
        PnL = (holding * delta_price).sum(dim=(1,2))
        C_T = self.loss_fn.terminal_payoff(S_att[:,-1])
        X = -PnL + C_T
        p0 = find_optimal_p0(X,self.loss_fn.alpha)
        X = X - p0
        return S_att, V_att, VarPrice_att

    def net_W2_att_S(self,network, S,V,VarPrice,delta, ratio, n,q=2,):
        sequence_length = S.shape[1] - 1
        att = torch.zeros(S.shape[0], sequence_length+1).to(device)
        att.requires_grad = True
        alpha = delta*ratio/n
        for i in range(n):
            S_att = S+att
            input_vector = torch.cat((torch.log(S_att[:, :-1]).unsqueeze(-1), V[:, :-1].unsqueeze(-1)), dim=-1)
            holding = network(input_vector).squeeze()
            loss = self.loss_fn(holding, S_att, VarPrice)
            
            grad = torch.autograd.grad(loss, att)[0]*S.shape[0]
            grad[:,0] = 0
            grad_norm = grad.norm(p=1,dim=1, keepdim=True)
            #att
            att = att + alpha*torch.sign(grad)*grad_norm.pow(q-1)*((grad_norm.pow(q).mean()+1e-10).pow(1/q-1))
            #proj
            dist = att.norm(p=float('inf'), dim=1, keepdim=True)
            p = (1-1/q)**(-1)
            r = min(1.0, delta/(dist.pow(p).mean().pow(1/q)+1e-10))
            att = torch.clamp(att, -r*dist, r*dist)
            att[:,0] = 0
            att = torch.clamp(att,-S)

        S_att = S + att

        input_vector = torch.cat((torch.log(S_att[:, :-1]).unsqueeze(-1), V[:, :-1].unsqueeze(-1)), dim=-1)
        holding = network(input_vector).squeeze()
        delta_S = S_att[:, 1:] - S_att[:, :-1]
        delta_V = VarPrice[:, 1:] - VarPrice[:, :-1]
        delta_price = torch.cat((delta_S.unsqueeze(-1), delta_V.unsqueeze(-1)), dim=2)
        PnL = (holding * delta_price).sum(dim=(1,2))
        C_T = self.loss_fn.terminal_payoff(S_att[:,-1])
        X = -PnL + C_T
        p0 = find_optimal_p0(X,self.loss_fn.alpha)
        X = X - p0
        return S_att, V, VarPrice

    def perfromance(self,network, S,V):
        VarPrice = self.V_to_VarPrice(V)
        input_vector = torch.cat((torch.log(S[:, :-1]).unsqueeze(-1), V[:, :-1].unsqueeze(-1)), dim=-1)
        holding = network(input_vector).squeeze()
        delta_S = S[:, 1:] - S[:, :-1]
        delta_V = VarPrice[:, 1:] - VarPrice[:, :-1]
        delta_price = torch.cat((delta_S.unsqueeze(-1), delta_V.unsqueeze(-1)), dim=2)
        PnL = (holding * delta_price).sum(dim=(1,2))
        C_T = self.loss_fn.terminal_payoff(S[:,-1])
        X = -PnL + C_T
        p0 = find_optimal_p0(X,self.loss_fn.alpha)
        X = X - p0
        return torch.max(X,torch.zeros_like(X)).mean().item()/(1-self.loss_fn.alpha)+p0, X, p0
    
    def S_budget_attack(self, network, S, V, delta,ratio,iter,return_att=False):
        if delta == 0:
            if return_att:
                return S, V, self.V_to_VarPrice(V), torch.zeros(S.shape[0], S.shape[1], 2).to(device)
            else:
                return S, V, self.V_to_VarPrice(V)
        #define parameters
        budget = torch.ones(S.shape[0], 2).to(device)*delta
        budget.requires_grad = True
        att_sign = torch.ones(S.shape[0], S.shape[1], 2).sign().to(device)
        att_sign_old = att_sign.clone().detach()
        att_sign[:,0,:] = 0
        att_sign.requires_grad = True
        #step size
        alpha = delta*ratio/iter
        #best statistics
        att_best = (budget.unsqueeze(1)*att_sign).clone().detach()
        result_best = 0.

        for iter in range(20):
            S_att = S + budget[:,0].unsqueeze(1)*att_sign[:, :, 0]
            V_att = V
            if self.perfromance(network, S_att, V_att)[0]>result_best:
                att_best = (budget.unsqueeze(1)*att_sign).clone().detach()
                result_best = self.perfromance(network, S_att, V_att)[0]
            VarPrice_att = self.V_to_VarPrice(V_att)
            input_vector = torch.cat((torch.log(S_att[:, :-1]).unsqueeze(-1), V_att[:, :-1].unsqueeze(-1)), dim=-1)
            holding = network(input_vector).squeeze()
            loss = self.loss_fn(holding, S_att, VarPrice_att)
            grad_b = torch.autograd.grad(loss, budget,retain_graph=True)[0]
            grad_a = torch.autograd.grad(loss, att_sign,retain_graph=True)[0]
            with torch.no_grad():
                # att
                budget_new = budget + alpha * grad_b.pow(2 - 1) * ((grad_b.pow(2).mean() + 1e-10).pow(1 / 2 - 1))
                # proj
                budget_new = budget_new / budget_new.square().mean().sqrt() * delta/1.414
                budget_new = torch.max(budget_new, torch.zeros_like(budget_new))
                budget.copy_(budget_new)
                # att_sign_new = (att_sign+1.0*0.75*grad_a.sign()+0.25*(att_sign-att_sign_old)).clamp(-1,1)
                # att_sign_new[:,0,:] = 0
                # att_sign_old = att_sign
                grad_a[:,0,:] = 0
                att_sign.copy_(grad_a.sign())
        S_att = S + budget[:,0].unsqueeze(1)*att_sign[:, :, 0]
        V_att = V 
        if self.perfromance(network, S_att, V_att)[0]>result_best:
            att_best = (budget.unsqueeze(1)*att_sign).clone().detach()
            result_best = self.perfromance(network, S_att, V_att)[0]
        
        S_att = S + att_best[:, :, 0]
        V_att = V
        VarPrice_att = self.V_to_VarPrice(V_att)
        if return_att:
            return S_att, V_att, VarPrice_att, att_best
        else:
            return S_att, V_att, VarPrice_att

    def SV_budget_attack(self, network, S, V, delta,ratio,iter,return_att=False):
        if delta == 0:
            if return_att:
                return S, V, self.V_to_VarPrice(V), torch.zeros(S.shape[0], S.shape[1], 2).to(device)
            else:
                return S, V, self.V_to_VarPrice(V)
        #define parameters
        budget = torch.ones(S.shape[0], 2).to(device)*delta
        budget.requires_grad = True
        att_sign = torch.ones(S.shape[0], S.shape[1], 2).sign().to(device)
        att_sign_old = att_sign.clone().detach()
        att_sign[:,0,:] = 0
        att_sign.requires_grad = True
        #step size
        alpha = delta*ratio/iter
        #best statistics
        att_best = (budget.unsqueeze(1)*att_sign).clone().detach()
        result_best = 0.

        for iter in range(20):
            S_att = S + budget[:,0].unsqueeze(1)*att_sign[:, :, 0]
            V_att = V + budget[:,1].unsqueeze(1)*att_sign[:, :, 1]/100
            if self.perfromance(network, S_att, V_att)[0]>result_best:
                att_best = (budget.unsqueeze(1)*att_sign).clone().detach()
                result_best = self.perfromance(network, S_att, V_att)[0]
            VarPrice_att = self.V_to_VarPrice(V_att)
            input_vector = torch.cat((torch.log(S_att[:, :-1]).unsqueeze(-1), V_att[:, :-1].unsqueeze(-1)), dim=-1)
            holding = network(input_vector).squeeze()
            loss = self.loss_fn(holding, S_att, VarPrice_att)
            grad_b = torch.autograd.grad(loss, budget,retain_graph=True)[0]
            grad_a = torch.autograd.grad(loss, att_sign,retain_graph=True)[0]
            with torch.no_grad():
                # att
                budget_new = budget + alpha * grad_b.pow(2 - 1) * ((grad_b.pow(2).mean() + 1e-10).pow(1 / 2 - 1))
                # proj
                budget_new = budget_new / budget_new.square().mean().sqrt() * delta/1.414
                budget_new = torch.max(budget_new, torch.zeros_like(budget_new))
                budget.copy_(budget_new)
                att_sign_new = (att_sign+1.0*0.75*grad_a.sign()+0.25*(att_sign-att_sign_old)).clamp(-1,1)
                att_sign_new[:,0,:] = 0
                att_sign_old = att_sign
                # grad_a[:,0,:] = 0
                # att_sign.copy_(grad_a.sign())
                att_sign.copy_(att_sign_new)
        S_att = S + budget[:,0].unsqueeze(1)*att_sign[:, :, 0]
        V_att = V + budget[:,1].unsqueeze(1)*att_sign[:, :, 1]/100 
        if self.perfromance(network, S_att, V_att)[0]>result_best:
            att_best = (budget.unsqueeze(1)*att_sign).clone().detach()
            result_best = self.perfromance(network, S_att, V_att)[0]
        
        S_att = S + att_best[:, :, 0]
        V_att = V + att_best[:, :, 1]/100
        VarPrice_att = self.V_to_VarPrice(V_att)
        if return_att:
            return S_att, V_att, VarPrice_att, att_best
        else:
            return S_att, V_att, VarPrice_att