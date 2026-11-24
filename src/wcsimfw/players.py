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
those with a value (for "sofifa": the highest overall). Every feature is centred on the
tournament's mean team, so it reads "better than the average side here".
"""

from __future__ import annotations

from datetime import date

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

    def agg(df: pl.DataFrame, prefix: str) -> pl.DataFrame:
        cols = [pl.col("value").mean().alias(prefix + "q")]
        cols += [pl.col("top_share").mean().alias(prefix + "top")] if top else []
        return df.group_by("tournament", "team").agg(cols)

    if split:
        d = pl.col("group").is_in(["GK", "DEF"])
        t = agg(xi.filter(~d), "att_").join(agg(xi.filter(d), "def_"), on=["tournament", "team"])
    else:
        t = agg(xi, "")
    cols = [c for c in t.columns if c not in ("tournament", "team")]
    return t.with_columns([(pl.col(c) - pl.col(c).mean().over("tournament")) for c in cols])


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


def design(table: pl.DataFrame, features: pl.DataFrame | None = None
           ) -> tuple[pl.DataFrame, dict[str, torch.Tensor], list[str]]:
    """Fit inputs for `match.fit_poisson` from `with_elo` matches and `team_features`. Without
    `split`, each rate gets own - opponent for every feature; with `split`, a rate gets the
    scoring side's attack features and minus the other side's defence features. Returns the
    matches kept (both teams have features), the tensors and the feature names."""
    t, names, xh, xa = table, [], None, None
    if features is not None:
        cols = [c for c in features.columns if c not in ("tournament", "team")]
        for side in ("home", "away"):
            f = features.rename({c: f"{side}_{c}" for c in cols} | {"team": side})
            t = t.join(f, on=["tournament", side], how="inner")
        att = [c for c in cols if not c.startswith("def_")]
        dfn = [c for c in cols if c.startswith("def_")]

        def col(name):
            return torch.tensor(t[name].to_numpy(), dtype=torch.float64)

        if dfn:
            xh = torch.stack([col(f"home_{c}") for c in att] + [-col(f"away_{c}") for c in dfn], 1)
            xa = torch.stack([col(f"away_{c}") for c in att] + [-col(f"home_{c}") for c in dfn], 1)
        else:
            xh = torch.stack([col(f"home_{c}") - col(f"away_{c}") for c in att], 1)
            xa = -xh
        names = att + dfn
    x = {k: torch.tensor(t[c].cast(pl.Float64).to_numpy(), dtype=torch.float64) for k, c in
         (("gap", "gap"), ("home", "at_home"), ("goals_home", "goals_home"),
          ("goals_away", "goals_away"))}
    return t, x | {"x_home": xh, "x_away": xa}, names
