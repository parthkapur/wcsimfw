"""Match models: the chance of every scoreline between two teams, as a score matrix.

Football sentence: each side scores at its own steady rate through the match, so a stronger side
scores more often and concedes less; a level knockout tie goes to thirty more minutes, then
penalties.
Pipeline place: match (strength -> match -> tournament). Every model returns a score matrix
`[..., G, G]`: entry [i, j] is P(home scores i, away scores j) over 90 minutes, batched over pairs,
with G = 11 (0 to 10 goals; the mass above 10 goes into the last row and column).
Numbers: `MU` (goals per team per match between equal sides) and `SLOPE` (how fast the rate
grows with the Elo gap) are set by hand by default; `fit_poisson` measures them (with a home
term and any team features) by Poisson maximum likelihood on past matches, with a standard error
from the curvature of the likelihood (the spread of a flat-prior Bayesian posterior). Extra
time plays at `EXTRA_TIME` of the 90-minute rate (30 of 90 minutes); the shootout is a coin
(`SHOOTOUT`), because shootout skill is weak evidence. All are arguments, so a run config can
change them.
"""

from __future__ import annotations

import torch
from torch import Tensor

GOALS = 11
MU = 1.30
SLOPE = 0.54
EXTRA_TIME = 1 / 3
SHOOTOUT = 0.5


def poisson(rate_home: Tensor, rate_away: Tensor, goals: int = GOALS) -> Tensor:
    """Score matrix of independent Poisson goals at the two rates."""
    rate_home, rate_away = torch.broadcast_tensors(torch.as_tensor(rate_home),
                                                   torch.as_tensor(rate_away))
    k = torch.arange(goals, dtype=rate_home.dtype, device=rate_home.device)

    def pmf(rate: Tensor) -> Tensor:
        r = rate[..., None]
        p = torch.exp(torch.xlogy(k, r) - r - torch.lgamma(k + 1))[..., :-1]
        return torch.cat([p, (1 - p.sum(-1, keepdim=True)).clamp_min(0)], -1)

    return pmf(rate_home)[..., :, None] * pmf(rate_away)[..., None, :]


def elo_rates(elo_home: Tensor, elo_away: Tensor, mu: float = MU,
              slope: float = SLOPE) -> tuple[Tensor, Tensor]:
    """Goal rates from World Football Elo: log rate = log mu + slope * (own - other) / 400.

    No home term: at a World Cup most sides play on neutral ground.
    """
    gap = slope * (torch.as_tensor(elo_home) - torch.as_tensor(elo_away)) / 400
    return mu * torch.exp(gap), mu * torch.exp(-gap)


def elo_poisson(elo_home: Tensor, elo_away: Tensor, mu: float = MU, slope: float = SLOPE,
                goals: int = GOALS) -> Tensor:
    """The Elo reference model: Poisson goals at the Elo rates."""
    return poisson(*elo_rates(elo_home, elo_away, mu, slope), goals)


def outcome(matrix: Tensor) -> Tensor:
    """[..., 3]: P(home win), P(draw), P(away win) from a score matrix."""
    return torch.stack([
        torch.tril(matrix, -1).sum((-2, -1)),
        torch.diagonal(matrix, dim1=-2, dim2=-1).sum(-1),
        torch.triu(matrix, 1).sum((-2, -1)),
    ], -1)


def advance(full: Tensor, extra: Tensor, shootout: float = SHOOTOUT) -> Tensor:
    """P(home goes through a knockout tie): win in 90 minutes, else in extra time, else on
    penalties. `full` is the 90-minute matrix, `extra` the 30-minute one (for Poisson models,
    the same model at `EXTRA_TIME` of each rate); `shootout` is P(home wins the shootout)."""
    f, e = outcome(full), outcome(extra)
    return f[..., 0] + f[..., 1] * (e[..., 0] + e[..., 1] * shootout)


def elo_pairs(elo: Tensor, mu: float = MU, slope: float = SLOPE, extra_time: float = EXTRA_TIME,
              shootout: float = SHOOTOUT, attack: Tensor | None = None,
              defence: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Tournament inputs for teams with Elo `elo` [T]: the score matrix of every pair
    [T, T, G, G] (row team at home) and P(row team goes through a knockout tie) [T, T].
    `attack` and `defence` [T] (log-rate offsets from team features, e.g. `players`): a side's
    rate grows with its attack and falls with the other side's defence."""
    rh, ra = elo_rates(elo[:, None], elo[None, :], mu, slope)
    if attack is not None:
        rh = rh * torch.exp(attack[:, None] - defence[None, :])
        ra = ra * torch.exp(attack[None, :] - defence[:, None])
    full = poisson(rh, ra)
    return full, advance(full, poisson(rh * extra_time, ra * extra_time), shootout)


def poisson_nll(theta: Tensor, gap: Tensor, home: Tensor, goals_home: Tensor,
                goals_away: Tensor, x_home: Tensor | None = None,
                x_away: Tensor | None = None) -> Tensor:
    """Per-match negative log-likelihood of both scores (constant log k! terms left out).

    log rate home = log mu + slope * gap / 400 + home_term * home + x_home @ beta
    log rate away = log mu - slope * gap / 400 + x_away @ beta
    with theta = (log mu, slope, home_term, beta...)."""
    lh = theta[0] + theta[1] * gap / 400 + theta[2] * home
    la = theta[0] - theta[1] * gap / 400
    if x_home is not None:
        lh, la = lh + x_home @ theta[3:], la + x_away @ theta[3:]
    return lh.exp() - goals_home * lh + la.exp() - goals_away * la


def fit_poisson(gap: Tensor, home: Tensor, goals_home: Tensor, goals_away: Tensor,
                x_home: Tensor | None = None, x_away: Tensor | None = None
                ) -> tuple[Tensor, Tensor]:
    """Maximum-likelihood theta of `poisson_nll` and its covariance (inverse curvature: the
    spread of a flat-prior Bayesian posterior). Terms the data cannot measure (no home matches)
    stay at 0 with nan error."""
    args = [torch.as_tensor(v, dtype=torch.float64) for v in (gap, home, goals_home, goals_away)]
    xs = [None if x is None else torch.as_tensor(x, dtype=torch.float64) for x in (x_home, x_away)]
    k = 0 if xs[0] is None else xs[0].shape[1]

    def nll(th):
        return poisson_nll(th, *args, *xs).sum()

    th = torch.zeros(3 + k, dtype=torch.float64)
    th[:2] = torch.tensor([0.25, 0.5])
    th.requires_grad_(True)
    opt = torch.optim.LBFGS([th], max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = nll(th)
        loss.backward()
        return loss

    opt.step(closure)
    th = th.detach()
    idx = torch.tensor([0, 1] + [2] * bool(args[1].any()) + list(range(3, 3 + k)))
    cov = torch.full((3 + k, 3 + k), float("nan"), dtype=torch.float64)
    cov[idx[:, None], idx] = torch.autograd.functional.hessian(nll, th)[idx][:, idx].inverse()
    return th, cov


def fit_elo_poisson(gap: Tensor, home: Tensor, goals_home: Tensor, goals_away: Tensor,
                    x_home: Tensor | None = None, x_away: Tensor | None = None,
                    names: list[str] | None = None) -> dict[str, tuple[float, float]]:
    """`fit_poisson` as {"mu", "slope", "home_term", *names}: (value, standard error)."""
    th, cov = fit_poisson(gap, home, goals_home, goals_away, x_home, x_away)
    se = cov.diag().sqrt().tolist()
    mu = th[0].exp().item()
    out = {"mu": (mu, mu * se[0]), "slope": (th[1].item(), se[1]),
           "home_term": (th[2].item(), se[2])}
    return out | {n: (th[3 + i].item(), se[3 + i]) for i, n in enumerate(names or [])}
