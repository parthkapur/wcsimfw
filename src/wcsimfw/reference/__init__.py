"""Reference facts of a tournament: format, teams, played results and a dated Elo table.

Football sentence: the rules of the competition and what happened on the pitch, written
down once so that every model is played and scored against the same facts.
Pipeline place: data (fixed facts shipped with the package). Nothing here is fitted.

Files per tournament (`reference/<tournament>/`):
- `format.json`: groups, how teams qualify, ranking criteria, knockout matches by FIFA
  number (`1E` winner of group E, `2C` runner-up, `3ABCDF` best third from those groups,
  `W74` winner of match 74, `L101` loser), and `annex_c`: for each set of eight groups whose
  thirds qualify, which group's third meets which group winner.
- `teams.json`: group, confederation, host, final group position, rank among the thirds,
  FIFA ranking at the draw and the release used for the tie-break.
- `results.json`: every match; goals after 90 minutes, after extra time, penalties.
- `elo.json`: World Football Elo ratings on a date before kickoff.
Each file names the raw files it was built from, with their sha256.
"""

from __future__ import annotations

import json
from importlib.resources import files


def load(name: str, tournament: str = "wc2026") -> dict:
    """Read one reference file, e.g. load("format") or load("results", "wc2026")."""
    return json.loads(files(__name__).joinpath(tournament, f"{name}.json").read_text("utf-8"))
