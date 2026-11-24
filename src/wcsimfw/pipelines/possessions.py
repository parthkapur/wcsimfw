"""Possessions pipeline: count how national teams moved the ball, possession by possession.

Football sentence: a possession is a run of touches by one side; it moves between its own third,
the middle third and the final third, and ends with a shot (scored or missed) or a lost ball.
Pipeline place: data -> processed artifact (input to the possession chain). Nothing is fitted.

Input: openplay event files (Opta events from WhoScored) of national-team tournaments and the
openplay match table. Only the two halves of normal time are used (the chain models 90 minutes).
States of a touch: B (x < 1/3 of the pitch, own third), M (middle), F (final third), S (shot).
A possession is a run of touches by one team; it ends at its first shot or when the other side
touches the ball. Own goals and penalty shootouts are not part of it.

Outputs (`$WCSIMFW_DATA/processed/possessions/`, Parquet + manifest.json):
- `moves`: tournament, match_id, team, from (B, M, F, S), to (B, M, F, S, loss, goal, miss), n.
- `starts`: tournament, match_id, team, state (state of the first touch), n.
- `matches`: tournament, match_id, date, home, away, goals_home, goals_away (after 90 minutes),
  possessions_home, possessions_away.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl

from wcsimfw import data
from wcsimfw.pipelines.players import team

TOURNAMENTS = ("afcon_2015", "copa_2015", "euro_2016", "wc_2018", "afcon_2019", "copa_2019",
               "euro_2020", "afcon_2021", "wc_2022", "afcon_2023", "copa_2024", "euro_2024",
               "afcon_2025", "wc_2026")
OP = "raw/openplay/data-v0.1.0"


def touches(events: pl.DataFrame) -> pl.DataFrame:
    """Touches of normal time in possession order, with `poss`, `state` and `to`."""
    e = (events.with_row_index("row")
         .filter(pl.col("is_touch") & pl.col("period").cast(str).is_in(["FirstHalf", "SecondHalf"]))
         .sort("match_id", "row"))
    e = e.with_columns((pl.col("team") != pl.col("team").shift(1).over("match_id"))
                       .fill_null(True).cum_sum().over("match_id").alias("poss"))
    third = 100 / 3
    e = e.with_columns(pl.when(pl.col("is_shot")).then(pl.lit("S"))
                       .when(pl.col("x") < third).then(pl.lit("B"))
                       .when(pl.col("x") < 2 * third).then(pl.lit("M"))
                       .otherwise(pl.lit("F")).alias("state"))
    shots_before = (pl.col("state") == "S").cast(pl.Int32).cum_sum().over("match_id", "poss")
    e = e.filter((shots_before - (pl.col("state") == "S").cast(pl.Int32)) == 0)  # stop at a shot
    nxt = pl.col("state").shift(-1).over("match_id", "poss")
    return e.with_columns(
        pl.when(pl.col("state") == "S")
        .then(pl.when(pl.col("is_goal")).then(pl.lit("goal")).otherwise(pl.lit("miss")))
        .otherwise(nxt.fill_null("loss")).alias("to"))


def build(raw: Path | None = None, out: Path | None = None) -> Path:
    if raw is None:
        raw = data.fetch([f"{OP}/events_{t}.parquet" for t in TOURNAMENTS]
                         + [f"{OP}/matches.parquet"]) / "raw"
    out = out or data.data_dir() / "processed/possessions"
    out.mkdir(parents=True, exist_ok=True)
    cols = ["match_id", "period", "team", "x", "is_touch", "is_shot", "is_goal"]
    info = pl.read_parquet(raw / "openplay/data-v0.1.0/matches.parquet")
    moves, starts, matches = [], [], []
    for t in TOURNAMENTS:
        e = touches(pl.read_parquet(raw / f"openplay/data-v0.1.0/events_{t}.parquet",
                                    columns=cols)).with_columns(pl.col("team").cast(str))
        tag = pl.lit(t).alias("tournament")
        moves.append(e.group_by("match_id", "team", pl.col("state").alias("from"), "to")
                     .len("n").with_columns(tag))
        first = e.group_by("match_id", "poss", maintain_order=True).first()
        starts.append(first.group_by("match_id", "team", "state").len("n").with_columns(tag))
        per = dict(((m, tm), n) for m, tm, n in
                   first.group_by("match_id", "team").len().iter_rows())
        m = info.filter(pl.col("match_id").is_in(e["match_id"].unique().implode()))
        matches += [{"tournament": t, "match_id": r["match_id"], "date": str(r["date"])[:10],
                     "home": team(r["home_team"]), "away": team(r["away_team"]),
                     "goals_home": r["home_score_90"], "goals_away": r["away_score_90"],
                     "possessions_home": per.get((r["match_id"], r["home_team"]), 0),
                     "possessions_away": per.get((r["match_id"], r["away_team"]), 0)}
                    for r in m.iter_rows(named=True)]
    tables = {"moves": pl.concat(moves), "starts": pl.concat(starts),
              "matches": pl.DataFrame(matches)}
    for k in ("moves", "starts"):
        tables[k] = tables[k].with_columns(pl.col("team").map_elements(team, pl.String))
    manifest = {"tournaments": TOURNAMENTS, "data_revision": data.REVISION, "files": {},
                "rows": {k: v.height for k, v in tables.items()}}
    for k, df in tables.items():
        df.write_parquet(out / f"{k}.parquet")
        manifest["files"][k] = hashlib.sha256((out / f"{k}.parquet").read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out


def load(name: str) -> pl.DataFrame:
    """One output table, built first if missing."""
    path = data.data_dir() / "processed/possessions" / f"{name}.parquet"
    if not path.exists():
        build()
    return pl.read_parquet(path)


if __name__ == "__main__":
    print(build())
