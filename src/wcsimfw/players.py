"""Player strength: from the registered squad's players to an attack and a defence per team.

Football sentence: a side is as good as the eleven of its 26 who play: a player's season match
rating says how well he played, his minutes how sure we can be of it, his league how hard the
football was, and keepers and defenders stop goals while midfielders and forwards make them.
Pipeline place: strength (processed players table -> strength -> match).
Numbers: per player, nothing is fitted; the weights that turn features into goals are fitted by
`match.fit_poisson`. Settings (one per DESIGN feature, so each can be switched off):
- `source`: "rating" (season match rating) or "sofifa" (SoFIFA overall, dated before kickoff).
- `tau`: minutes reliability. A player's rating is pulled toward his position's mean as if
  `tau` more minutes at that mean were added: (m r + tau r0) / (m + tau). This is the
  Bayesian posterior mean of a rating with a prior worth `tau` minutes. 0 = off.
- `top`: add the share of the eleven's minutes in the five big leagues and the Champions
  League as a feature (its weight is the competition shift).
- `split`: attack (midfielders, forwards) and defence (keeper, defenders) as separate features.
- `halflife`: days; with "rating", rate from pre-tournament club form matches, each weighted by
  2^(-age / halflife) (recency). None = the season rating.
The eleven: the goalkeeper and the ten outfield players with the most season minutes among
those with a value (for "sofifa": the highest overall); a side with fewer than eleven such
players has no features. Every feature is centred on the tournament's mean team, so it reads
"better than the average side here".

Elo already knows most of it: good squads win matches. `residuals` keeps only what the rating
does not say ("this squad is better than its results"), so the squad corrects the rating and
cannot replace it; a side without features is as good as its rating says.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import torch

from wcsimfw import elo


def player_values(players: pl.DataFrame, form: pl.DataFrame | None = None, source: str = "rating",
                  tau: float = 0.0, halflife: float | None = None) -> pl.DataFrame:
    """One value per player (`value`), plus `minutes`, `top_share` and the position group."""
    p = players.with_columns(
        (pl.col("top_minutes") / pl.col("minutes").clip(lower_bound=1)).alias("top_share"),
        pl.col("position").replace({"Goalkeeper": "GK", "Defender": "DEF", "Midfielder": "MID",
                                    "Attacker": "FWD"}).alias("group"))
    if source == "sofifa":
        return p.with_columns(pl.col("sofifa").cast(pl.Float64).alias("value"))
    if halflife is not None:
        k = p.select("tournament", "kickoff").unique()
        f = form.join(k, on="tournament").with_columns(
            (2 ** (-(pl.col("kickoff").str.to_date() - pl.col("date").str.to_date())
                   .dt.total_days() / halflife) * pl.col("minutes")).alias("w"))
        r = f.group_by("tournament", "player_id").agg(
            ((pl.col("w") * pl.col("rating")).sum() / pl.col("w").sum()).alias("r"),
            pl.col("minutes").sum().alias("m"))
        p = p.join(r, on=["tournament", "player_id"], how="left").with_columns(
            pl.col("r").alias("rating"), pl.col("m").fill_null(0).alias("rated_minutes"))
    else:
        p = p.with_columns(pl.col("minutes").alias("rated_minutes"))
    prior = pl.col("rating").mean().over("tournament", "group")
    m = pl.col("rated_minutes")
    return p.with_columns(((m * pl.col("rating") + tau * prior) / (m + tau)).alias("value")
                          if tau else pl.col("rating").alias("value"))


def eleven(values: pl.DataFrame, source: str = "rating") -> pl.DataFrame:
    """The goalkeeper and ten outfield players per team (see module text)."""
    key = "value" if source == "sofifa" else "minutes"
    v = values.filter(pl.col("value").is_not_null()).with_columns(
        (pl.col("group") == "GK").alias("gk"))
    ranked = v.sort(key, descending=True).with_columns(
        pl.int_range(pl.len()).over("tournament", "team", "gk").alias("k"))
    return ranked.filter(pl.when(pl.col("gk")).then(pl.col("k") < 1).otherwise(pl.col("k") < 10))


def team_features(players: pl.DataFrame, form: pl.DataFrame | None = None,
                  source: str = "rating", tau: float = 0.0, top: bool = False,
                  split: bool = False, halflife: float | None = None) -> pl.DataFrame:
    """Per tournament and team: feature columns, centred. Without `split`: `q` (and `top`);
    with `split`: `att_q`, `def_q` (and `att_top`, `def_top`)."""
    xi = eleven(player_values(players, form, source, tau, halflife), source)
    xi = xi.filter(pl.len().over("tournament", "team") == 11)

    def agg(df: pl.DataFrame, prefix: str) -> pl.DataFrame:
        cols = [pl.col("value").mean().alias(prefix + "q")]
        cols += [pl.col("top_share").mean().alias(prefix + "top")] if top else []
        return df.group_by("tournament", "team").agg(cols)

    if split:
        d = pl.col("group").is_in(["GK", "DEF"])
        t = agg(xi.filter(~d), "att_").join(agg(xi.filter(d), "def_"), on=["tournament", "team"])
    else:
        t = agg(xi, "")
    return t.with_columns([(pl.col(c) - pl.col(c).mean().over("tournament"))
                           for c in _columns(t)])


def _columns(features: pl.DataFrame) -> list[str]:
    return [c for c in features.columns if c not in ("tournament", "team")]


def residuals(features: pl.DataFrame, kickoff: dict[str, str], history: list[dict],
              use: list[str]) -> pl.DataFrame:
    """Features less what each side's rating at kickoff predicts of them: a straight line through
    the (centred) ratings, measured on the tournaments `use`."""
    ratings: dict[str, float] = {}
    at = {}
    todo = sorted(kickoff, key=kickoff.get)
    for h in [*history, {"date": "9999"}]:
        while todo and h["date"] >= kickoff[todo[0]]:
            at[todo.pop(0)] = dict(ratings)
        if todo:
            elo.update(ratings, h)
    f = features.with_columns(pl.Series("elo", [
        at[t].get(x, elo.START) for t, x in features.select("tournament", "team").iter_rows()]))
    f = f.with_columns(pl.col("elo") - pl.col("elo").mean().over("tournament"))
    fit = f.filter(pl.col("tournament").is_in(use))
    slope = {c: (fit[c] * fit["elo"]).sum() / max((fit["elo"] ** 2).sum(), 1e-9)
             for c in _columns(features)}
    return f.with_columns([pl.col(c) - slope[c] * pl.col("elo") for c in slope]).drop("elo")


def covariates(rows: list[dict], features: pl.DataFrame, matches: pl.DataFrame
               ) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str | None]]:
    """(x_home, x_away) [N, K] for `match.fit_poisson` on history matches `rows` (date, home,
    away): own - opponent for every feature, zero where a side has none; a feature named def_*
    counts for the side it belongs to when it defends. Also the feature names and, per row, the
    tournament of `matches` (processed `players` table) it belongs to, else None."""
    names = _columns(features)
    k = [n.startswith("def_") for n in names]
    table = {(t, x): v for t, x, *v in features.select("tournament", "team", *names).iter_rows()}
    none = [0.0] * len(names)
    span = [(t, *(date.fromisoformat(d) for d in (a, b)), set(h) | set(w)) for t, a, b, h, w in
            matches.group_by("tournament").agg(
                pl.col("date").min().alias("a"), pl.col("date").max().alias("b"),
                pl.col("home"), pl.col("away")).iter_rows()]
    xh, xa, tour = [], [], []
    for r in rows:
        day = date.fromisoformat(r["date"])
        t = next((t for t, a, b, teams in span if a - timedelta(1) <= day <= b + timedelta(1)
                  and r["home"] in teams and r["away"] in teams), None)
        h, a = table.get((t, r["home"]), none), table.get((t, r["away"]), none)
        xh.append([(-y if d else x) if any(k) else x - y
                   for x, y, d in zip(h, a, k, strict=True)])
        xa.append([(-x if d else y) if any(k) else y - x
                   for x, y, d in zip(h, a, k, strict=True)])
        tour.append(t)
    as_t = [torch.tensor(x, dtype=torch.float64).reshape(len(rows), len(names)) for x in (xh, xa)]
    return *as_t, names, tour


def with_elo(matches: pl.DataFrame) -> pl.DataFrame:
    """Each match with its pre-match Elo gap (home - away) from the results history, turned to
    the history's home side, and `home` = 1 if that side played at home. A match is found by its
    two teams within one day (time zones); matches not found are dropped."""
    rows = matches.to_dicts()
    want: dict[frozenset, list[int]] = {}
    for i, m in enumerate(rows):
        want.setdefault(frozenset((m["home"], m["away"])), []).append(i)
    ratings: dict[str, float] = {}
    out = []
    for h in elo.history(max(m["date"] for m in rows) + "~"):
        gap = elo.update(ratings, h)
        for i in want.get(frozenset((h["home"], h["away"])), []):
            m = rows[i]
            if abs((date.fromisoformat(m["date"]) - date.fromisoformat(h["date"])).days) <= 1:
                same = h["home"] == m["home"]
                g = (m["goals_home"], m["goals_away"]) if same else (m["goals_away"],
                                                                     m["goals_home"])
                out.append(m | {"home": h["home"], "away": h["away"], "goals_home": g[0],
                                "goals_away": g[1], "gap": gap, "at_home": not h["neutral"]})
    return pl.DataFrame(out)
