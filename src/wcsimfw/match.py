"""Match models: the chance of every scoreline between two teams, as a score matrix.

Football sentence: each side scores at its own steady rate through the match, so a stronger side
scores more often and concedes less; a level knockout tie goes to thirty more minutes, then
penalties.
Pipeline place: match (strength -> match -> tournament). Every model returns a score matrix
`[..., G, G]`: entry [i, j] is P(home scores i, away scores j) over 90 minutes, batched over pairs,
with G = 11 (0 to 10 goals; the mass above 10 goes into the last row and column).
Numbers: nothing here is fitted. `MU` (goals per team per match between equal sides) and
`SLOPE` (how fast the rate grows with the Elo gap) are fixed, set by hand; a fit to the 2018
and 2022 World Cups gives nearly the same slope. Extra time plays at `EXTRA_TIME` of the
90-minute rate (30 of 90 minutes); the shootout is a coin (`SHOOTOUT`), because shootout skill
is weak evidence. All are arguments, so a run config can change them.
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
              shootout: float = SHOOTOUT) -> tuple[Tensor, Tensor]:
    """Tournament inputs for teams with Elo `elo` [T]: the score matrix of every pair
    [T, T, G, G] (row team at home) and P(row team goes through a knockout tie) [T, T]."""
    rh, ra = elo_rates(elo[:, None], elo[None, :], mu, slope)
    full = poisson(rh, ra)
    return full, advance(full, poisson(rh * extra_time, ra * extra_time), shootout)
