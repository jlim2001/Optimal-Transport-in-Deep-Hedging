"""
First-order Wasserstein sensitivity penalty -- the drop-in replacement for the
WBPGD adversarial inner loop in Heston_train_adv.py.

Idea (Bartl, Drapeau, Oblój & Wiesel, sensitivity analysis of Wasserstein DRO):
    sup_{Q in B_delta(P)} E_Q[l]  =  E_P[l]  +  delta * Upsilon  +  o(delta)
so the expensive adversarial loss  l(x_adv)  is replaced by  l(x) + delta * Upsilon.

Upsilon must be computed for the SAME Wasserstein ball the authors' attacker uses,
otherwise the comparison is unfair. Reading Heston_Attacker.SV_budget_attack:

  * perturbation of path n, coordinate c in {S, V}:  Delta_{n,c} = b_{n,c} * sign_{n,c,t}
    -> per-path cost is the sup-norm over time  ||Delta_{n,c}||_inf = b_{n,c}
  * budgets are normalised so  sqrt(mean_{n,c} b^2) = delta / sqrt(2),
    i.e.  mean_n  sum_c ||Delta_{n,c}||_inf^2  <=  delta^2     (a W2-type ball)
  * the V coordinate is perturbed in units of 1/100:  V_att = V + att_V / 100
  * t = 0 is never perturbed (all paths share S0, v0)
  * the variance-swap price is RECOMPUTED from the perturbed V (V_to_VarPrice)

Dual of that ball (Cauchy-Schwarz + Hoelder, l_inf <-> l_1 over time):
    Upsilon = sqrt( mean_n  sum_c  || grad_{x_{n,c}} l_n ||_1^2 ),
with the l_1 norm taken over t = 1..T, and the V-gradient taken w.r.t. the
attack coordinate att_V (i.e. grad_V / 100).

S-attack: only c = S is perturbed. (Caveat: the authors' S_budget_attack keeps an
unused second budget column in the normalisation, so its effective S-radius lies
between delta/sqrt(2) and delta. We use delta, the upper end; see README_FORK.md.)
"""
import torch


def upsilon(network, loss_fn, attacker, S, V, p0, attack_method="SV",
            create_graph=True):
    """
    Returns (Upsilon, loss_at_x) where loss_at_x is the batch-mean loss at the
    unperturbed paths (reused as the first term of the first-order expansion).

    network       : the hedging network (call in eval mode, as the authors do for
                    the adversarial branch, so per-path gradients are independent
                    of the rest of the batch -- train-mode BatchNorm would couple them)
    loss_fn       : the authors' loss (loss_CVAR with p0_mode='given'); any loss
                    with the same call signature works, so the base risk measure
                    can be swapped without touching this file
    attacker      : Heston_Attacker, used only for V_to_VarPrice (same mapping as attack)
    p0            : the p0 the adversarial branch uses (p0_att in their script)
    create_graph  : True so the penalty is differentiable w.r.t. theta (double backprop)
    """
    B = S.shape[0]
    sv = (attack_method == "SV")

    # Zero perturbations as leaves: gradients w.r.t. these equal the gradients
    # w.r.t. the attack coordinates, in exactly the units the attacker uses.
    dS = torch.zeros_like(S, requires_grad=True)
    S_in = S + dS
    if sv:
        dV = torch.zeros_like(V, requires_grad=True)
        V_in = V + dV / 100.0                    # attacker's V scaling
    else:
        V_in = V
    VarPrice_in = attacker.V_to_VarPrice(V_in)   # recomputed, as in the attack

    inp = torch.cat((torch.log(S_in[:, :-1]).unsqueeze(-1),
                     V_in[:, :-1].unsqueeze(-1)), dim=-1)
    holding = network(inp).squeeze()
    loss = loss_fn(holding, S_in, VarPrice_in, p0=p0)   # batch mean

    leaves = [dS, dV] if sv else [dS]
    grads = torch.autograd.grad(loss, leaves, create_graph=create_graph)

    # loss is a batch MEAN -> multiply by B for per-path gradients (authors do the same)
    sq = 0.0
    for g in grads:
        g = g[:, 1:] * B                         # drop t = 0 (never perturbed)
        sq = sq + g.abs().sum(dim=1).pow(2)      # ||.||_1 over time, squared
    ups = (sq.mean() + 1e-12).sqrt()
    return ups, loss
