"""Determinism: how much of a match, or of a tournament, is decided by who is better.

Football sentence: in some sports the better side nearly always wins; in football a single goal
can turn the match, so the cup often goes to somebody else, and these numbers say how often.
Pipeline place: evaluation (reads probabilities of matches and of tournaments; any sport).
Numbers: nothing is fitted.

- Match: `skill_share`, the part of the variance of results that the forecast explains, 0 (a coin)
  to 1 (the better side always wins). By the law of total variance, Var(y) = Var(p) + E[p(1 - p)]
  when the forecast p is calibrated: skill plus luck.
- Series: `series`, P(win a best-of-n), the other way to play more.
- Tournament: `entropy` of who wins it (bits), `contenders` = 2^entropy (as open as a cup between
  that many equal sides), and `tournament_share` = 1 - entropy / log2(teams), 0 (all equal) to 1
  (known before it starts).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def skill_share(p: Tensor, y: Tensor) -> tuple[float, float]:
    """(skill share, standard error over matches) of forecasts p of results y (1 won, 0.5 level,
    0 lost): 1 - E[(y - p)^2] / Var(y)."""
    p, y = torch.as_tensor(p).double(), torch.as_tensor(y).double()
    var = y.var(unbiased=False)
    d = 1 - (y - p) ** 2 / var
    return d.mean().item(), (d.std() / len(d) ** 0.5).item()


def series(p: Tensor | float, n: int) -> Tensor | float:
    """P(win a best-of-n series) at P(win one match) p, n odd, matches independent."""
    k = n // 2 + 1
    return sum(math.comb(k - 1 + i, i) * p**k * (1 - p) ** i for i in range(k))


def entropy(p: Tensor) -> Tensor:
    """Entropy in bits of distributions over the last axis."""
    return -torch.special.xlogy(p, p).sum(-1) / math.log(2)


def contenders(p: Tensor) -> Tensor:
    """Number of equal sides that would make the title as open."""
    return 2 ** entropy(p)


def tournament_share(p: Tensor) -> Tensor:
    """1 - entropy / log2(number of entrants)."""
    return 1 - entropy(p) / math.log2(p.shape[-1])
