"""Evaluation: score a forecast against what happened in the played tournament.

Football sentence: a good forecast gave high chances to the results that came and to the teams
that went far, and its 70% calls came true about 70% of the time.
Pipeline place: evaluation (tournament -> evaluation). Inputs are the score matrices and
knockout matrix of a match model; outputs are scores at one origin (kickoff or after a stage).
Numbers: nothing is fitted. All scores are "lower is better".

- Matches scored at an origin: every group match not yet played, and the knockout matches of
  the next round once its line-up is known. Each is scored on its 90-minute result
  (home win, draw, away win) with the probabilities of the score matrix (exact, no draws).
  `match_rps`: ranked probability score (squared error of the cumulative probabilities, over
  2 terms); `brier`: squared error over the 3 outcomes, summed; `log_score`: -ln P(result);
  `match_rps_uniform`: the RPS of 1/3 each on the same matches, as a floor to beat.
- `stage_rps`: for each of the 48 teams, the mean squared error of P(reach r32, r16, qf, sf,
  final, champion) against 1 or 0, then the mean over teams. Stages already decided at the
  origin stay in (they score 0), so compare models only at the same origin.
  `stage_rps_se` is its Monte Carlo error.
- `calibration`: forecasts put in bins of width 1/bins; per bin the count, mean forecast and
  how often the event happened.
"""

from __future__ import annotations

import torch
from torch import Tensor

from wcsimfw.match import outcome
from wcsimfw.tournament import ORDER, STAGES, Tournament, simulate


def reached(t: Tournament) -> Tensor:
    """[T, 6] 1 where the team reached the stage (r32 ... champion) in the played tournament."""
    idx = {team: i for i, team in enumerate(t.teams)}
    out = torch.zeros(len(t.teams), len(STAGES))
    for m in t.results:
        if m["stage"] in STAGES:
            out[[idx[m["home"]], idx[m["away"]]], STAGES.index(m["stage"])] = 1
    out[t.held("final")[1][104], -1] = 1
    return out


def targets(t: Tournament, after: str | None) -> list[dict]:
    """Matches to score at an origin: unplayed group matches, and the next knockout round."""
    last = -1 if after is None else ORDER.index(after)
    idx = {team: i for i, team in enumerate(t.teams)}
    out = []
    for m in t.results:
        stage = f"md{m['matchday']}" if m["stage"] == "group" else m["stage"]
        k = ORDER.index(stage) if stage in ORDER else -1
        if k > last and (m["stage"] == "group" or k == last + 1):
            h, a = m["goals_90"]
            out.append({"stage": stage, "home": idx[m["home"]], "away": idx[m["away"]],
                        "result": 0 if h > a else 1 if h == a else 2})
    return out


def match_scores(p: Tensor, y: Tensor) -> dict[str, Tensor]:
    """Per-match RPS, Brier and log score of outcome probabilities p [N, 3] for results y [N]."""
    hot = torch.nn.functional.one_hot(y, 3).to(p.dtype)
    cum = (p.cumsum(-1) - hot.cumsum(-1))[..., :2]
    return {"match_rps": (cum**2).sum(-1) / 2, "brier": ((p - hot) ** 2).sum(-1),
            "log_score": -p.gather(-1, y[:, None]).squeeze(-1).log()}


def stage_rps(p: Tensor, truth: Tensor) -> Tensor:
    """Per-team stage RPS [T] of P(reach stage) p [T, 6] against the stages reached."""
    return ((p - truth) ** 2).mean(-1)


def calibration(p: Tensor, y: Tensor, bins: int = 10) -> Tensor:
    """[bins, 3]: count, mean forecast, observed rate of forecasts p (any shape) of events y."""
    p, y = p.flatten().double(), y.flatten().double()
    b = (p * bins).long().clamp(0, bins - 1)
    n = torch.bincount(b, minlength=bins).double()
    s = [torch.zeros(bins, dtype=torch.float64).index_add_(0, b, x) for x in (p, y)]
    return torch.stack([n, s[0] / n, s[1] / n], -1)  # empty bins give nan


def evaluate(t: Tournament, full: Tensor, adv: Tensor, after: str | None = None,
             n: int = 100_000, seed: int = 0, chunk: int = 50_000) -> dict:
    """Simulate from an origin and score it. Returns `stages` (the `simulate` output), `matches`
    (the scored matches with `p` [N, 3]), `scores` (floats) and `calibration` ("match", "stage")."""
    sim = simulate(t, full, adv, n, seed, after, chunk)
    p, se = sim["exact"], sim["exact_se"]
    truth = reached(t).to(p)
    # ponytail: delta method, covariance between teams ignored; a bootstrap over draws if needed
    stage_se = ((2 * (p - truth) / p.numel() * se) ** 2).sum().sqrt()
    games = targets(t, after)
    pm = outcome(full[[g["home"] for g in games], [g["away"] for g in games]])
    y = torch.tensor([g["result"] for g in games], device=pm.device)
    per = match_scores(pm, y)
    scores = {k: v.mean().item() for k, v in per.items()}
    scores |= {"matches": len(games),
               "match_rps_uniform": match_scores(torch.full_like(pm, 1 / 3), y)["match_rps"]
               .mean().item(),
               "stage_rps": stage_rps(p, truth).mean().item(), "stage_rps_se": stage_se.item()}
    open_ = (p > 0) & (p < 1)  # stages not yet decided at the origin
    return {
        "stages": sim, "matches": [g | {"p": q} for g, q in zip(games, pm.tolist(), strict=True)],
        "scores": scores,
        "calibration": {"match": calibration(pm, torch.nn.functional.one_hot(y, 3)),
                        "stage": calibration(p[open_], truth[open_])},
    }
