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
