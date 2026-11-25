"""Possession chain: goals as the end of possessions that move up the pitch (a Markov chain).

Football sentence: a side keeps the ball through its own third, the middle and the final third,
takes a shot and scores or misses, or loses the ball on the way; a stronger side (Elo gap) moves
forward more often, loses it less and finishes more, and that is where its extra goals come from.
Pipeline place: match (strength -> match). Same interface as the other match models: a score
matrix from two goal rates. Input: the Elo gap; the processed `possessions` tables for the fit.
Numbers: fitted, by maximum likelihood on the counted moves of earlier tournaments:
- from each zone (B own third, M middle, F final third): P(next = B, M, F, shot, loss) as a
  multinomial logit in the Elo gap (a + b * gap / 400), 4 + 4 numbers per zone;
- shot: P(goal) as a logistic in the gap; first touch of a possession: P(B, M, F, shot).
- `n`: possessions per team in 90 minutes, and `own_goals`: all goals / goals from the chain's
  shots (own goals are goals too); both measured on the same matches (`constants`).
P(goal | possession) comes from the fundamental matrix (I - Q)^-1 of the absorbing chain (the
expected visits to each zone); the goal rate is n * own_goals * P(goal | possession), Poisson.
- `home`: log factor on the goals of a side at home, from the same matches (`residuals`).
Set by hand: the zone borders (thirds of the pitch) and `prior` of `style`.

Team style (`style`): a side plays as its rating says until its own matches show otherwise. Per
zone and next step, its counted moves O are compared with the moves E the chain expected in the
same matches; the factor (K + O) / (K + E) is the posterior mean of a Gamma(K, K) factor on
Poisson counts, with K worth `prior` matches. It is measured for a side's own moves and for the
moves it allows. Finishing (goal per shot) has no team factor: past finishing does not predict.
"""

from __future__ import annotations

import polars as pl
import torch
from torch import Tensor

from wcsimfw import match

ZONES = ("B", "M", "F")
NEXT = ("B", "M", "F", "S", "loss")  # "loss" is the reference category
FIRST = ("B", "M", "F", "S")  # "B" is the reference category


def _unpack(theta: Tensor) -> dict[str, Tensor]:
    i = iter(range(len(theta)))

    def take(*shape):
        n = 1
        for s in shape:
            n *= s
        return theta[[next(i) for _ in range(n)]].reshape(shape)

    return {"move_a": take(3, 4), "move_b": take(3, 4), "shot_a": take(1), "shot_b": take(1),
            "first_a": take(3), "first_b": take(3)}


SIZE = 3 * 4 * 2 + 2 + 3 * 2


def probabilities(theta: Tensor, gap: Tensor) -> dict[str, Tensor]:
    """Chain probabilities for Elo gaps `gap` [...]: `move` [..., 3, 5], `goal` [...] (per shot),
    `first` [..., 4]."""
    p = _unpack(theta)
    g = torch.as_tensor(gap, dtype=theta.dtype)[..., None, None] / 400
    logits = p["move_a"] + p["move_b"] * g
    move = torch.softmax(torch.cat([logits, torch.zeros_like(logits[..., :1])], -1), -1)
    goal = torch.sigmoid(p["shot_a"][0] + p["shot_b"][0] * g[..., 0, 0])
    f = p["first_a"] + p["first_b"] * g[..., 0, :]
    first = torch.softmax(torch.cat([torch.zeros_like(f[..., :1]), f], -1), -1)
    return {"move": move, "goal": goal, "first": first}


def goal_per_possession(theta: Tensor, gap: Tensor, style: dict[str, Tensor] | None = None
                        ) -> Tensor:
    """P(a possession ends in a goal) for a side `gap` Elo points above its opponent. With
    `style`, `gap` is [T, T] (row team against column team, in the order of the style)."""
    p = probabilities(theta, gap)
    if style is not None:
        for k in ("first", "move"):
            x = p[k] * style[f"{k}_for"][:, None] * style[f"{k}_against"][None, :]
            p[k] = x / x.sum(-1, keepdim=True)
    q = p["move"][..., :3]
    shoot = p["move"][..., 3] * p["goal"][..., None]
    eye = torch.eye(3, dtype=q.dtype, device=q.device)
    h = torch.linalg.solve(eye - q, shoot[..., None])[..., 0]  # P(goal | in each zone)
    return (p["first"][..., :3] * h).sum(-1) + p["first"][..., 3] * p["goal"]


def nll(theta: Tensor, moves: dict[str, Tensor], firsts: dict[str, Tensor]) -> Tensor:
    """Negative log-likelihood of counted moves and first touches (see `counts`)."""
    pm = probabilities(theta, moves["gap"])
    zone = moves["from"].clamp(max=2)
    move_p = pm["move"][torch.arange(len(zone)), zone, moves["to"].clamp(max=4)]
    shot = moves["from"] == 3
    goal_p = torch.where(moves["to"] == 5, pm["goal"], 1 - pm["goal"])
    p = torch.where(shot, goal_p, move_p)
    pf = probabilities(theta, firsts["gap"])["first"]
    q = pf[torch.arange(len(firsts["state"])), firsts["state"]]
    return -(moves["n"] * p.log()).sum() - (firsts["n"] * q.log()).sum()


def counts(moves: pl.DataFrame, starts: pl.DataFrame, gaps: pl.DataFrame
           ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Tensors for `nll` from the processed tables and a gap per (match_id, team)."""
    code = {s: i for i, s in enumerate(NEXT)} | {"goal": 5, "miss": 6}

    def t(df, c):
        return torch.tensor(df[c].to_numpy())

    m = moves.join(gaps, on=["match_id", "team"]).with_columns(
        pl.col("from").replace_strict({"B": 0, "M": 1, "F": 2, "S": 3}).alias("f"),
        pl.col("to").replace_strict(code).alias("t"))
    s = starts.join(gaps, on=["match_id", "team"]).with_columns(
        pl.col("state").replace_strict({k: i for i, k in enumerate(FIRST)}).alias("s"))
    return ({"gap": t(m, "gap").double(), "from": t(m, "f"), "to": t(m, "t"),
             "n": t(m, "n").double()},
            {"gap": t(s, "gap").double(), "state": t(s, "s"), "n": t(s, "n").double()})


def team_gaps(matches: pl.DataFrame) -> pl.DataFrame:
    """(match_id, team, gap): each side's pre-match Elo gap, from `players.with_elo` rows."""
    return pl.concat([matches.select("match_id", pl.col("home").alias("team"), "gap"),
                      matches.select("match_id", pl.col("away").alias("team"), -pl.col("gap"))])


def constants(matches: pl.DataFrame, moves: pl.DataFrame) -> tuple[float, float]:
    """(n, own_goals) measured on `matches` (processed possessions tables)."""
    m = moves.filter(pl.col("match_id").is_in(matches["match_id"].implode()))
    n = (matches["possessions_home"] + matches["possessions_away"]).mean() / 2
    goals = (matches["goals_home"] + matches["goals_away"]).sum()
    return n, goals / m.filter(pl.col("to") == "goal")["n"].sum()


def fit(moves: dict[str, Tensor], firsts: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
    """Maximum-likelihood theta and its covariance (inverse curvature)."""
    th = torch.zeros(SIZE, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([th], max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = nll(th, moves, firsts)
        loss.backward()
        return loss

    opt.step(closure)
    th = th.detach()
    return th, torch.autograd.functional.hessian(lambda x: nll(x, moves, firsts), th).inverse()


def style(theta: Tensor, moves: pl.DataFrame, starts: pl.DataFrame, matches: pl.DataFrame,
          teams: list[str], prior: float) -> dict[str, Tensor]:
    """Team factors (see module text) from the moves of `matches` (`players.with_elo` rows):
    `first_for`, `first_against` [T, 4] and `move_for`, `move_against` [T, 3, 5]. A team without
    matches gets 1."""
    flip = {"home": "away", "away": "home"}
    sides = pl.concat([matches.select(
        "match_id", pl.col(a).alias("team"), pl.col(b).alias("other"),
        (pl.col("gap") if a == "home" else -pl.col("gap")).alias("gap"))
        for a, b in flip.items()]).with_row_index("side")
    p = probabilities(theta, torch.tensor(sides["gap"].to_numpy()))
    obs = {"first": torch.zeros_like(p["first"]), "move": torch.zeros_like(p["move"])}
    key = ["match_id", "team"]
    for r in starts.join(sides, on=key).iter_rows(named=True):
        obs["first"][r["side"], FIRST.index(r["state"])] += r["n"]
    for r in moves.filter(pl.col("from") != "S").join(sides, on=key).iter_rows(named=True):
        obs["move"][r["side"], ZONES.index(r["from"]), NEXT.index(r["to"])] += r["n"]
    out = {}
    for k, o in obs.items():
        total = o.sum(-1, keepdim=True)
        weight = (prior * total.mean(0)).clamp_min(1e-9)
        for name, col in (("for", "team"), ("against", "other")):
            at = [teams.index(x) if x in teams else len(teams) for x in sides[col]]
            so, se = (weight + torch.zeros(len(teams) + 1, *o.shape[1:], dtype=o.dtype).index_add(
                0, torch.tensor(at, dtype=torch.long), x) for x in (o, total * p[k]))
            out[f"{k}_{name}"] = (so / se)[:-1]
    return out


def residuals(theta: Tensor, matches: pl.DataFrame, scale: float) -> pl.DataFrame:
    """`matches` with the goals the chain expects of each side (`e_home`, `e_away`), at the rate
    `scale` (n * own_goals) per possession."""
    gap = torch.tensor(matches["gap"].to_numpy())
    return matches.with_columns(
        pl.Series("e_home", (scale * goal_per_possession(theta, gap)).numpy()),
        pl.Series("e_away", (scale * goal_per_possession(theta, -gap)).numpy()))


def rates(elo: Tensor, theta: Tensor, n: float, own_goals: float = 1.0,
          style: dict[str, Tensor] | None = None) -> Tensor:
    """Goal rate [T, T] of the row team against the column team:
    n * own_goals * P(goal | possession, gap, style)."""
    gap = elo[:, None] - elo[None, :]
    style = style and {k: v.to(elo) for k, v in style.items()}
    return n * own_goals * goal_per_possession(theta.to(elo), gap, style)


def pairs(elo: Tensor, theta: Tensor, n: float, own_goals: float = 1.0,
          extra_time: float = match.EXTRA_TIME, shootout: float = match.SHOOTOUT,
          style: dict[str, Tensor] | None = None, host: Tensor | None = None
          ) -> tuple[Tensor, Tensor]:
    """`match.pairs` of `rates`: tournament inputs, like `match.elo_pairs`."""
    return match.pairs(rates(elo, theta, n, own_goals, style), extra_time, shootout, host)
