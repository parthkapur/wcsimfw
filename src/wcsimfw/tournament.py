"""Tournament: play the whole World Cup many times at once and count how far each team goes.

Football sentence: every group match is played from its score matrix, the tables are ranked by
the competition's rules, the qualifiers are placed in the bracket, and each knockout tie is won in
90 minutes, in extra time or on penalties; matches already played keep their real result.
Pipeline place: tournament (match -> tournament -> evaluation). Inputs are a score matrix for
every pair of teams `[T, T, G, G]` and the knockout matrix `[T, T]` (P(row team goes through
against column team)), both from a match model.
Numbers: nothing is fitted. The format (groups, ranking criteria, bracket, FIFA Annexe C) comes
from the reference data. Team conduct (cards) is not simulated; the FIFA ranking release named
`tiebreak` in the reference data decides any tie that reaches it.

Two knockout methods use the same bracket: `knockout_mc` draws every tie; `knockout_exact`
carries, for each match, the probability of each team winning it, so the chance of reaching each
round given a round-of-32 line-up is exact. Averaged over the same simulated line-ups, the two
must agree within Monte Carlo error; the exact one has the smaller error.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from wcsimfw.reference import load

STAGES = ("r32", "r16", "qf", "sf", "final", "champion")
# Stage of a played match, in order; an origin "after X" holds every match up to X at its result.
ORDER = ("md1", "md2", "md3", "r32", "r16", "qf", "sf", "final")
_ROUND = {"r32": range(73, 89), "r16": range(89, 97), "qf": range(97, 101),
          "sf": range(101, 103), "final": (104,)}


@dataclass
class Tournament:
    teams: list[str]  # group-major: team index = 4 * group + place in the sorted group
    groups: str
    fixtures: Tensor  # [M, 2] team indices of the group matches
    fifa_rank: Tensor  # [T]
    bracket: dict[int, tuple[str, str]]  # match number -> two refs ("1E", "3ABCDF", "W74")
    annex: Tensor  # [4096, n winners]: group of the third that meets each listed winner
    annex_winners: str
    results: list[dict]

    @classmethod
    def load(cls, name: str = "wc2026") -> Tournament:
        fmt, tm, res = (load(n, name) for n in ("format", "teams", "results"))
        groups = "".join(fmt["groups"])
        teams = sorted(tm["teams"], key=lambda t: (tm["teams"][t]["group"], t))
        idx = {t: i for i, t in enumerate(teams)}
        ranks = tm["fifa_ranking"]["releases"][tm["fifa_ranking"]["tiebreak"]]
        winners = "".join(next(iter(fmt["annex_c"].values())))
        annex = torch.full((1 << len(groups), len(winners)), -1, dtype=torch.long)
        for key, row in fmt["annex_c"].items():
            mask = sum(1 << groups.index(g) for g in key)
            annex[mask] = torch.tensor([groups.index(row[w]) for w in winners])
        group = [m for m in res["matches"] if m["stage"] == "group"]
        return cls(
            teams=teams, groups=groups,
            fixtures=torch.tensor([[idx[m["home"]], idx[m["away"]]] for m in group]),
            fifa_rank=torch.tensor([ranks[t]["rank"] for t in teams]),
            bracket={int(k): tuple(v) for k, v in fmt["knockout"].items() if int(k) != 103},
            annex=annex, annex_winners=winners, results=res["matches"],
        )

    def held(self, after: str | None) -> tuple[dict[int, tuple[int, int]], dict[int, int]]:
        """Played results held fixed at an origin: group fixture -> goals, match -> winner.
        `after` is None (kickoff) or a stage in ORDER, e.g. "md2" or "r16"."""
        if after is None:
            return {}, {}
        last = ORDER.index(after)
        idx = {t: i for i, t in enumerate(self.teams)}
        pairs = {tuple(f): i for i, f in enumerate(self.fixtures.tolist())}
        goals, wins = {}, {}
        for m in self.results:
            stage = f"md{m['matchday']}" if m["stage"] == "group" else m["stage"]
            if stage not in ORDER or ORDER.index(stage) > last:
                continue
            h, a = idx[m["home"]], idx[m["away"]]
            if m["stage"] == "group":
                goals[pairs[h, a]] = tuple(m["goals_90"])
            else:
                g = m["penalties"] or m["goals_120"] or m["goals_90"]
                wins[m["match"]] = h if g[0] > g[1] else a
        return goals, wins


def _lexsort_desc(keys: list[Tensor]) -> Tensor:
    """Indices that sort the last axis by keys (most significant first), largest first."""
    order = torch.arange(keys[0].shape[-1], device=keys[0].device).expand_as(keys[0])
    for k in reversed(keys):
        order = order.gather(-1, k.gather(-1, order).argsort(dim=-1, descending=True, stable=True))
    return order


def play_groups(t: Tournament, matrix: Tensor, n: int, gen: torch.Generator,
                held: dict[int, tuple[int, int]] | None = None) -> Tensor:
    """[n, M, 2] goals of every group match; held matches keep their real score."""
    f = t.fixtures.to(matrix.device)
    p = matrix[f[:, 0], f[:, 1]].flatten(-2)  # [M, G*G]
    draw = torch.multinomial(p, n, replacement=True, generator=gen).T  # [n, M]
    goals = torch.stack([draw // matrix.shape[-1], draw % matrix.shape[-1]], -1)
    for i, g in (held or {}).items():
        goals[:, i] = torch.tensor(g, device=goals.device)
    return goals


def rank_groups(t: Tournament, goals: Tensor) -> tuple[Tensor, Tensor]:
    """Art. 13 tables. Returns positions [n, groups, 4] (team indices, first to fourth) and the
    groups of the qualified thirds as a bit mask [n]."""
    n, dev = goals.shape[0], goals.device
    ng = len(t.groups)
    f = t.fixtures.to(dev)
    g, i, j = f[:, 0] // 4, f[:, 0] % 4, f[:, 1] % 4
    gf = torch.zeros(n, ng * 16, dtype=goals.dtype, device=dev)  # goals of row team vs column
    gf[:, g * 16 + i * 4 + j] = goals[..., 0]
    gf[:, g * 16 + j * 4 + i] = goals[..., 1]
    gf = gf.view(n, ng, 4, 4)
    off = ~torch.eye(4, dtype=torch.bool, device=dev)
    diff = gf - gf.transpose(-1, -2)
    pts = (3 * (diff > 0) + (diff == 0)) * off
    total = [pts.sum(-1), diff.sum(-1), gf.sum(-1)]
    keys = [total[0]]
    same = (total[0][..., :, None] == total[0][..., None, :]) & off
    for _ in range(3):  # head-to-head among the teams level, repeated on any smaller level set
        h2h = [(x * same).sum(-1) for x in (pts, diff, gf)]
        keys += h2h
        for x in h2h:
            same = same & (x[..., :, None] == x[..., None, :])
    rank = t.fifa_rank.to(dev).view(ng, 4).expand(n, ng, 4)
    order = _lexsort_desc(keys + total[1:] + [-rank])
    base = torch.arange(ng, device=dev)[:, None] * 4
    pos = order + base
    third = order[..., 2:3]  # place of each group's third
    tk = [x.gather(-1, third).squeeze(-1) for x in total + [-rank]]
    best = _lexsort_desc(tk)[:, : 8]
    mask = (1 << best).sum(-1)
    return pos, mask


def line_up(t: Tournament, pos: Tensor, mask: Tensor) -> Tensor:
    """[n, 32] round-of-32 teams, two per match in match order (73, 74, ...)."""
    thirds = t.annex.to(pos.device)[mask]  # [n, winners]
    if (thirds < 0).any():
        raise ValueError("qualified thirds not in Annexe C")
    cols = []
    for m in _ROUND["r32"]:
        a, b = t.bracket[m]
        for ref, other in ((a, b), (b, a)):
            if ref[0] == "3":
                grp = thirds[:, t.annex_winners.index(other[1])]
                cols.append(pos[torch.arange(len(pos)), grp, 2])
            else:
                cols.append(pos[:, t.groups.index(ref[1]), int(ref[0]) - 1])
    return torch.stack(cols, -1)


def _bracket_order(t: Tournament):
    return [(m, t.bracket[m]) for r in ("r16", "qf", "sf", "final") for m in _ROUND[r]]


def _hold(winner: Tensor, a: Tensor, b: Tensor, m: int, wins: dict[int, int]) -> Tensor:
    if m not in wins:
        return winner
    w = torch.full_like(winner, wins[m])
    if not ((w == a) | (w == b)).all():
        raise ValueError(f"held winner of match {m} is not in that match")
    return w


def knockout_mc(t: Tournament, lineup: Tensor, adv: Tensor, gen: torch.Generator,
                wins: dict[int, int] | None = None) -> Tensor:
    """[n, T, 6] bool: the team reached each stage (r32 ... champion), each tie drawn once."""
    wins = wins or {}
    n, dev = lineup.shape[0], lineup.device
    reach = torch.zeros(n, len(t.teams), len(STAGES), dtype=torch.bool, device=dev)
    rows = torch.arange(n, device=dev)
    reach[rows[:, None], lineup, 0] = True
    team = {}

    def play(m, a, b, stage):
        u = torch.rand(n, generator=gen, device=dev)
        w = _hold(torch.where(u < adv[a, b], a, b), a, b, m, wins)
        team[m] = w
        reach[rows, w, stage] = True

    for k, m in enumerate(_ROUND["r32"]):
        play(m, lineup[:, 2 * k], lineup[:, 2 * k + 1], 1)
    for m, (ra, rb) in _bracket_order(t):
        stage = 1 + next(i for i, r in enumerate(("r32", "r16", "qf", "sf", "final"))
                         if m in _ROUND[r])
        play(m, team[int(ra[1:])], team[int(rb[1:])], stage)
    return reach


def knockout_exact(t: Tournament, lineup: Tensor, adv: Tensor,
                   wins: dict[int, int] | None = None) -> Tensor:
    """[n, T, 6] P(team reaches each stage) for each line-up, by recursion over the bracket:
    P(i wins a match) = P(i reaches it) * sum_j P(j is the opponent) * adv[i, j]."""
    wins = wins or {}
    n, nt = lineup.shape[0], len(t.teams)
    rows = torch.arange(n, device=lineup.device)
    out = torch.zeros(n, nt, len(STAGES), dtype=adv.dtype, device=lineup.device)
    out[rows[:, None], lineup, 0] = 1
    dist = {}

    def play(m, p, q, stage):
        w = p * (q @ adv.T) + q * (p @ adv.T)
        if m in wins:
            if not (p[:, wins[m]] + q[:, wins[m]] > 0.999).all():
                raise ValueError(f"held winner of match {m} is not in that match")
            w = torch.zeros_like(w)
            w[:, wins[m]] = 1
        dist[m] = w
        out[..., stage] += w

    def one(team):
        return torch.nn.functional.one_hot(team, nt).to(adv.dtype)

    for k, m in enumerate(_ROUND["r32"]):
        play(m, one(lineup[:, 2 * k]), one(lineup[:, 2 * k + 1]), 1)
    for m, (ra, rb) in _bracket_order(t):
        stage = 1 + next(i for i, r in enumerate(("r32", "r16", "qf", "sf", "final"))
                         if m in _ROUND[r])
        play(m, dist.pop(int(ra[1:])), dist.pop(int(rb[1:])), stage)
    return out


def simulate(t: Tournament, matrix: Tensor, adv: Tensor, n: int = 10_000, seed: int = 0,
             after: str | None = None, chunk: int = 50_000) -> dict[str, Tensor]:
    """Play the tournament n times from an origin, `chunk` at a time (memory). Returns, per team
    and stage [T, 6]: `mc` (all ties drawn), `exact` (knockout by recursion on the same
    line-ups), and the Monte Carlo standard error of each (`mc_se`, `exact_se`).
    P(reach a stage | left the group) is `exact[:, k] / exact[:, 0]`."""
    gen = torch.Generator(matrix.device).manual_seed(seed)
    goals_held, wins = t.held(after)
    sums = {k: torch.zeros(2, len(t.teams), len(STAGES), dtype=adv.dtype, device=adv.device)
            for k in ("mc", "exact")}
    for start in range(0, n, chunk):
        m = min(chunk, n - start)
        goals = play_groups(t, matrix, m, gen, goals_held)
        lineup = line_up(t, *rank_groups(t, goals))
        for k, x in (("mc", knockout_mc(t, lineup, adv, gen, wins).to(adv.dtype)),
                     ("exact", knockout_exact(t, lineup, adv, wins))):
            sums[k] += torch.stack([x.sum(0), (x * x).sum(0)])
    out = {}
    for k, (s1, s2) in sums.items():
        mean = s1 / n
        out[k] = mean
        out[f"{k}_se"] = ((s2 / n - mean**2).clamp_min(0) / (n - 1)).sqrt()
    return out
