"""
Penalized (Lagrangian) Wasserstein adversary for Heston deep hedging.

Constrained problem (He, Sutter & Gonon 2025, SV_budget_attack):
    sup_{Q : W_c(P,Q) <= delta^2}  E_Q[ l ]
Penalized problem (this fork; Sinha, Namkoong & Duchi 2018 form):
    sup_Q { E_Q[ l ] - lam * W_c(P,Q) }  =  E_P[ sup_{x'} { l(x') - lam * c(x, x') } ]

The right-hand side is a SEPARATE maximisation per path: no shared budget, no projection.

Geometry is kept IDENTICAL to the constrained attack, so the only change is
constraint -> penalty:
  * perturbation of path n, coordinate c in {S, V}:  D_{n,c,t} = b_{n,c} * s_{n,c,t},
    with b >= 0 and s in [-1, 1]  (the authors' budget x sign parameterisation)
  * transport cost of a path:  c_n = sum_c  max_t |D_{n,c,t}|^2
  * V perturbed in units of 1/100 (V_att = V + D_V / 100), t = 0 never perturbed,
    variance-swap price recomputed from the perturbed V.

Algorithm
  1. Initialise at the exact maximiser of the LINEARISED penalised problem:
         s = sign(g),  b_c = ||g_c||_1 / (2 lam)        (g = per-path input gradient)
  2. Refine for n_iter steps:
       - s: the authors' signed-gradient update with momentum, clipped to [-1, 1]
       - b: damped best response  b <- (1-kappa) b + kappa * (dl/db) / (2 lam m^2),
            m = max_t |s|. For a loss that is mu-strongly concave in b this converges
            whenever lam > -mu/2 (the Sinha et al. curvature condition); see
            README_FORK.md and the report appendix.
  3. Keep, PER PATH, the iterate with the best penalised objective l_n - lam c_n.

Realised radius  sqrt(mean_n c_n)  is reported so that penalised and constrained runs
can be placed on a common axis (the constrained attack's radius is delta by design).
"""
import torch

from Heston_util import Heston_Attacker, find_optimal_p0


class Heston_Penalized_Attacker(Heston_Attacker):
    """Same constructor as Heston_Attacker; adds penalized_attack()."""

    # ---------- per-path CVaR loss (Rockafellar-Uryasev) ----------
    def _per_path_loss(self, network, S_att, V_att, p0):
        """
        l_n = p0 + (X_n - p0)^+ / (1 - alpha),  X_n = payoff - PnL (+ costs).
        p0 = None -> batch quantile ('search' mode, exactly as the constrained attack's
        loss uses); otherwise a fixed float (e.g. the trained p0_att).
        """
        VarPrice_att = self.V_to_VarPrice(V_att)
        inp = torch.cat((torch.log(S_att[:, :-1].clamp(min=1e-6)).unsqueeze(-1),
                         V_att[:, :-1].unsqueeze(-1)), dim=-1)
        holding = network(inp).squeeze()
        dS = S_att[:, 1:] - S_att[:, :-1]
        dVP = VarPrice_att[:, 1:] - VarPrice_att[:, :-1]
        dprice = torch.cat((dS.unsqueeze(-1), dVP.unsqueeze(-1)), dim=2)
        X = self.loss_fn.terminal_payoff(S_att[:, -1]) - (holding * dprice).sum(dim=(1, 2))
        tc = self.loss_fn.transaction_cost_rate
        if tc > 0:
            price = torch.cat((S_att.unsqueeze(-1), VarPrice_att.unsqueeze(-1)), dim=2)
            X = X + (torch.diff(holding, dim=1, prepend=holding[:, :1]).abs()
                     * tc * price[:, :-1, :]).sum(dim=(1, 2))
        if p0 is None:
            p0 = find_optimal_p0(X.detach(), self.loss_fn.alpha)
        a = self.loss_fn.alpha
        return torch.clamp(X - p0, min=0.0) / (1.0 - a) + p0

    @staticmethod
    def _perturb(S, V, b, s, nC):
        D = b.unsqueeze(1) * s                        # (B, T+1, nC)
        S_att = S + D[:, :, 0]
        V_att = V + D[:, :, 1] / 100.0 if nC == 2 else V
        return S_att, V_att, D

    @staticmethod
    def transport_cost(D):
        """Per-path cost  sum_c max_t |D_{n,c,t}|^2   -> (B,)"""
        return (D.abs().amax(dim=1) ** 2).sum(dim=1)

    # ---------- the attack ----------
    def penalized_attack(self, network, S, V, lam, n_iter=20, attack_method="SV",
                         kappa=0.5, p0=None, b_max=None):
        """
        Returns S_att, V_att, VarPrice_att, info  with
        info = {"radius": sqrt(mean_n c_n), "objective": mean_n (l_n - lam c_n),
                "loss_gain": mean_n l_n(x_att) - mean_n l_n(x)}.

        lam    : penalty (> 0). Larger lam = weaker adversary (smaller realised radius).
        n_iter : refinement steps after the linearised initialisation (0 = first-order only).
        kappa  : damping of the budget best-response update, in (0, 1].
        p0     : None = batch-quantile threshold (matches the constrained attack);
                 float = fixed threshold (theoretically exact separability).
        b_max  : optional cap on each budget (guards against runaway when lam is small).
        Call with network.eval(), as the authors do for their attack.
        """
        if lam <= 0:
            raise ValueError("lam must be > 0")
        S, V = S.detach(), V.detach()
        B, T1 = S.shape
        nC = 2 if attack_method == "SV" else 1
        p0v = None if p0 is None else float(p0)

        # ---- 1. linearised initialisation (exact maximiser of the first-order problem)
        dS = torch.zeros_like(S, requires_grad=True)
        if nC == 2:
            dV = torch.zeros_like(V, requires_grad=True)
            ell0 = self._per_path_loss(network, S + dS, V + dV / 100.0, p0v)
            gS, gV = torch.autograd.grad(ell0.sum(), [dS, dV])
            g = torch.stack([gS, gV], dim=-1)
        else:
            ell0 = self._per_path_loss(network, S + dS, V, p0v)
            g = torch.autograd.grad(ell0.sum(), [dS])[0].unsqueeze(-1)
        g = g.detach()
        g[:, 0, :] = 0.0                                  # t = 0 is never perturbed
        clean_mean = ell0.detach().mean()

        s = g.sign().clone().requires_grad_(True)
        b0 = g.abs().sum(dim=1) / (2.0 * lam)
        if b_max is not None:
            b0 = b0.clamp(max=b_max)
        b = b0.clone().requires_grad_(True)
        s_old = s.detach().clone()

        best_J = torch.full((B,), -float("inf"), device=S.device)
        best_D = torch.zeros(B, T1, nC, device=S.device)
        best_l = torch.zeros(B, device=S.device)

        # ---- 2. refinement
        for it in range(n_iter + 1):
            S_att, V_att, D = self._perturb(S, V, b, s, nC)
            ell = self._per_path_loss(network, S_att, V_att, p0v)
            J = ell - lam * self.transport_cost(D)
            with torch.no_grad():                         # ---- 3. per-path best
                imp = J > best_J
                best_J = torch.where(imp, J, best_J)
                best_l = torch.where(imp, ell, best_l)
                best_D[imp] = D[imp]
            if it == n_iter:
                break
            gb, gs = torch.autograd.grad(ell.sum(), [b, s])
            with torch.no_grad():
                m2 = s.abs().amax(dim=1).pow(2).clamp(min=1e-12)
                b_new = ((1.0 - kappa) * b + kappa * gb / (2.0 * lam * m2)).clamp(min=0.0)
                if b_max is not None:
                    b_new = b_new.clamp(max=b_max)
                s_new = (s + 0.75 * gs.sign() + 0.25 * (s - s_old)).clamp(-1.0, 1.0)
                s_new[:, 0, :] = 0.0
                s_old = s.detach().clone()
                b.copy_(b_new)
                s.copy_(s_new)

        S_att = S + best_D[:, :, 0]
        V_att = V + best_D[:, :, 1] / 100.0 if nC == 2 else V
        VarPrice_att = self.V_to_VarPrice(V_att)
        info = {
            "radius": self.transport_cost(best_D).mean().sqrt().item(),
            "objective": best_J.mean().item(),
            "loss_gain": (best_l.mean() - clean_mean).item(),
        }
        return S_att.detach(), V_att.detach(), VarPrice_att.detach(), info


def realised_radius_from_att(att):
    """Radius of a perturbation in attack units, att: (B, T+1, nC) -> float.
    Use with SV_budget_attack(..., return_att=True) to put both methods on one axis."""
    return (att.abs().amax(dim=1) ** 2).sum(dim=1).mean().sqrt().item()
