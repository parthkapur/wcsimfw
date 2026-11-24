# wcsimfw

A small, fast World Cup simulation framework. Every match model gives a scoreline probability
matrix; the tournament plays the full format many times as batched tensors, on CPU or GPU with
the same code.

```text
data → strength → match (score matrix) → tournament → evaluation
```

## Install

```bash
uv sync            # Python 3.12; PyTorch runs on CPU on macOS, CUDA wheels on Linux
uv run pytest      # tests
uv run ruff check  # lint
```

## Data

`wcsimfw.data.fetch()` downloads a pinned revision of a Hugging Face dataset into a local cache.

| Variable | Meaning | Default |
|---|---|---|
| `WCSIMFW_DATA_REPO` | dataset id | `parthkapur/wcsimfw-data` (private) |
| `WCSIMFW_DATA_REVISION` | dataset commit to read | `wcsimfw.data.REVISION` |
| `WCSIMFW_DATA` | local cache folder | `~/.cache/wcsimfw` |

The dataset is laid out as `raw/<source>/...`, each source with a `SOURCE.md` (origin, dates,
licence) and a `MANIFEST.sha256`.

## Reference data

`wcsimfw.reference.load(name, tournament="wc2026")` returns the facts shipped with the package:

| Name | Content |
|---|---|
| `format` | groups, qualification, ranking criteria, knockout matches 73–104, FIFA Annexe C |
| `teams` | group, confederation, host, final group position, FIFA ranking releases |
| `results` | all 104 matches: goals after 90 minutes, after extra time, penalties |
| `elo` | World Football Elo ratings of 2026-06-04 (before kickoff) |

Each file records the raw files it was built from (path and sha256 in the dataset).

## Match models

Every model in `wcsimfw.match` returns a score matrix `[..., 11, 11]`: entry `[i, j]` is the
probability that home scores `i` and away scores `j` in 90 minutes, batched over pairs.

```python
import torch
from wcsimfw import match
from wcsimfw.reference import load

elo = load("elo")["ratings"]
m = match.elo_poisson(torch.tensor(elo["Spain"]), torch.tensor(elo["Argentina"]))
match.outcome(m)                       # P(home win), P(draw), P(away win)
rh, ra = match.elo_rates(torch.tensor(elo["Spain"]), torch.tensor(elo["Argentina"]))
x = match.EXTRA_TIME
match.advance(m, match.poisson(rh * x, ra * x))   # P(home goes through a knockout tie)
```

| Constant | Value | Meaning |
|---|---|---|
| `MU` | 1.30 | goals per team per match between equal sides (fixed) |
| `SLOPE` | 0.54 | log goal rate per 400 Elo points of gap (fixed) |
| `EXTRA_TIME` | 1/3 | extra-time goal rate as a share of the 90-minute rate (fixed) |
| `SHOOTOUT` | 0.5 | P(home wins a shootout) (fixed) |

## Tournament

`wcsimfw.tournament` plays the whole tournament `n` times as batched tensors: group matches from
the score matrices, tables ranked by the competition rules (points, then head-to-head, then goal
difference and goals; the FIFA ranking stands in for team conduct), the best thirds placed by
Annexe C, then the knockout. Matches already played keep their real result.

```python
import torch
from wcsimfw import match, tournament
from wcsimfw.reference import load

t = tournament.Tournament.load("wc2026")
elo = torch.tensor([load("elo")["ratings"][team] for team in t.teams], dtype=torch.float64)
full, adv = match.elo_pairs(elo)                    # all pairs: score matrix, P(goes through)
out = tournament.simulate(t, full, adv, n=100_000, after="md2")  # origin: after matchday 2
out["exact"]        # [48, 6] P(reach r32, r16, qf, sf, final, champion), order of t.teams
out["exact_se"]     # Monte Carlo standard error
```

Origins (`after`): `None` (kickoff), `md1`, `md2`, `md3`, `r32`, `r16`, `qf`, `sf`, `final`.
`mc` draws every knockout tie; `exact` computes the knockout by recursion over the bracket for each
simulated round-of-32 line-up. Both use the same line-ups and agree within Monte Carlo error;
`exact` has the smaller error. `exact[:, k] / exact[:, 0]` is P(reach stage k | left the group).

## Elo strength

`wcsimfw.elo` plays World Football Elo (eloratings.net rules: start 1500, home +100, K by
competition, goal-difference factor, shootout = draw) forward over a results history.

```python
from wcsimfw import elo
r = elo.ratings(elo.history("2026-06-11"))                    # every result before kickoff
r = elo.ratings(elo.tournament_matches(t.played("md2")), r)    # plus the played World Cup matches
```

`match.fit_elo_poisson(gap, home, goals_home, goals_away)` measures `mu`, `slope` and a home term
by Poisson maximum likelihood, each with a standard error.

## Players

`wcsimfw.pipelines.players` builds three processed tables from the raw data
(`$WCSIMFW_DATA/processed/players/`, with a manifest): each tournament's registered squads with
every player's season match rating, minutes, minutes in the five big leagues and the Champions
League, and SoFIFA overall (all from before kickoff); club form matches before kickoff; and the
tournament matches.

```bash
uv run python -m wcsimfw.pipelines.players   # build (also built on first use)
```

`wcsimfw.players` turns a squad into team features: the eleven (goalkeeper and ten outfield
players), each player's value from `source` ("rating" or "sofifa"), with optional minutes prior
(`tau`), competition share (`top`), attack/defence split (`split`) and recency (`halflife`).
`players.design` and `match.fit_poisson` fit the weights; in the match model each team gets an
attack and a defence (`match.elo_pairs(..., attack=, defence=)`).

## Possession chain

`wcsimfw.pipelines.possessions` counts, from national-team event data, how each possession moved:
own third (B), middle (M), final third (F), shot (S), and how it ended (goal, miss, lost ball).
`wcsimfw.chain` fits these moves as an absorbing Markov chain whose probabilities depend on the
Elo gap, and gives P(goal | possession) from the fundamental matrix (I - Q)^-1.

```python
from wcsimfw import chain
p = chain.goal_per_possession(theta, gap=torch.tensor(200.0))  # a side 200 Elo points better
full, adv = chain.pairs(elo, theta, n=179.3, own_goals=1.04)   # same outputs as elo_pairs
```

Model `possession_chain` in a run config fits the chain on the tournaments before kickoff
(`configs/possession_chain.toml`).

## Evaluation and run configs

`wcsimfw.evaluate` scores a forecast from one origin against the played tournament (all scores:
lower is better).

| Score | On what |
|---|---|
| `match_rps`, `brier`, `log_score` | 90-minute result (home win, draw, away win) of every group match not yet played and of the next knockout round once its line-up is known; exact probabilities from the score matrix |
| `match_rps_uniform` | the same matches forecast at 1/3 each |
| `stage_rps` | per team, mean squared error of P(reach r32, r16, qf, sf, final, champion) against what happened; mean over teams. Stages already decided stay in, so compare models at the same origin only |
| `*_se` | standard error over matches or teams: how far the score could move with other matches of the same kind. This decides whether a difference between models means anything |
| `stage_rps_mc_se` | Monte Carlo error of `stage_rps` (draws); far smaller than `stage_rps_se` |
| `calibration` | per probability bin: count, mean forecast, observed rate and z (gap in standard deviations if the forecasts were right). Read it at one origin |

Late origins have few open matches (8 after the group stage, 1 before the final); their `*_se`
shows that. Compare models mainly at kickoff and after matchdays 1 and 2.

```python
from wcsimfw import evaluate
e = evaluate.evaluate(t, full, adv, after="md1", n=100_000)
e["scores"]          # dict of the scores above
```

A run is one TOML file: tournament, origins, draws, seed, device, the model and its parameters,
each marked `fitted` or `fixed` with its `source`. See `configs/` (`elo_poisson.toml`: the dated
Elo table with hand-set numbers; `elo_fitted.toml`: Elo from the results history, mu and slope
fitted on World Cup and continental finals before kickoff; `elo_players.toml`: history Elo plus
the registered squads' SoFIFA eleven, fitted on earlier tournaments; `possession_chain.toml`:
the possession chain; the last three need the data).

```bash
uv run python -m wcsimfw.run configs/elo_poisson.toml   # writes runs/elo_poisson-<time>/
uv run python -m wcsimfw.run compare runs/A runs/B      # B - A per origin, paired, with errors
```

The run folder holds the config, `meta.json` (git commit, sha256 of each input file, the data
revision when the data were read, fitted values with standard errors, versions, device, seconds
per origin) and Parquet tables `scores`, `stages`, `matches`, `calibration`.
