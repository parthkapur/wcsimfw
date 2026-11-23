"""Elo strength: World Football Elo ratings, played forward match by match up to an origin.

Football sentence: a side gains points when it does better than its rating said it would, more
for a big win and more in a match that matters.
Pipeline place: strength (data -> strength -> match). Input: a results history (date, teams,
goals after extra time, competition, neutral ground); output: one rating per team.
Numbers: nothing is fitted; the rules are eloratings.net's, set by hand there. Every team starts
at 1500; home side +100; `K` by competition (`k_factor`); goal-difference factor 1, 1.5, then
(11 + n) / 8. A shootout counts as a draw. Only rating gaps matter to a match model.

Reading as a filter: a rating is the current belief about a side, and each result moves it by
K x (result - expected result). K says how much one match counts against everything before it:
a fixed-gain Bayesian update, with no data after the origin (it is a Markov chain in time).
"""

from __future__ import annotations

import csv

from wcsimfw import data

START = 1500
HOME = 100
FINALS = {"FIFA World Cup": 60, "UEFA Euro": 50, "Copa América": 50, "African Cup of Nations": 50,
          "AFC Asian Cup": 50, "Gold Cup": 50, "Confederations Cup": 50}
# Names in the results history -> names in the reference data (FIFA's).
ALIAS = {"Ivory Coast": "Côte d'Ivoire", "Turkey": "Türkiye", "Iran": "IR Iran",
         "DR Congo": "Congo DR", "United States": "USA", "South Korea": "Korea Republic",
         "Cape Verde": "Cabo Verde", "Czech Republic": "Czechia"}


def k_factor(competition: str) -> int:
    """60 World Cup finals, 50 continental finals, 40 qualifiers and Nations League,
    30 other tournaments, 20 friendlies."""
    if competition in FINALS:
        return FINALS[competition]
    if "qualification" in competition or "Nations League" in competition:
        return 40
    return 20 if competition == "Friendly" else 30


def expected(gap: float) -> float:
    """Expected result (win 1, draw 1/2) of a side `gap` rating points above its opponent."""
    return 1 / (10 ** (-gap / 400) + 1)


def update(ratings: dict[str, float], m: dict) -> float:
    """Play one match into `ratings` (in place). Returns the pre-match gap home - away
    (without the home bonus). `m`: home, away, goals (after extra time), competition, neutral."""
    h, a = m["home"], m["away"]
    rh, ra = ratings.get(h, START), ratings.get(a, START)
    gh, ga = m["goals"]
    n = abs(gh - ga)
    g = 1 if n <= 1 else 1.5 if n == 2 else (11 + n) / 8
    w = 1 if gh > ga else 0.5 if gh == ga else 0
    d = k_factor(m["competition"]) * g * (w - expected(rh - ra + (0 if m["neutral"] else HOME)))
    ratings[h], ratings[a] = rh + d, ra - d
    return rh - ra


def history(before: str) -> list[dict]:
    """International results (raw/international_results) dated before `before` (YYYY-MM-DD)."""
    path = data.fetch(["raw/international_results/*"]) / "raw/international_results/results.csv"
    with open(path, encoding="utf-8") as f:
        return [{"date": r["date"], "home": ALIAS.get(r["home_team"], r["home_team"]),
                 "away": ALIAS.get(r["away_team"], r["away_team"]),
                 "goals": (int(r["home_score"]), int(r["away_score"])),
                 "competition": r["tournament"], "neutral": r["neutral"] == "TRUE"}
                for r in csv.DictReader(f)
                if r["date"] < before and r["home_score"] not in ("", "NA")]


def tournament_matches(played: list[dict], competition: str = "FIFA World Cup") -> list[dict]:
    """Reference results (Tournament.played) as history rows: goals after extra time.
    ponytail: every match on neutral ground (hosts' home matches too); add HOME for hosts
    if a host term enters the match model."""
    return [{"date": m["date"], "home": m["home"], "away": m["away"],
             "goals": tuple(m["goals_120"] or m["goals_90"]), "competition": competition,
             "neutral": True} for m in played]


def ratings(matches: list[dict], start: dict[str, float] | None = None) -> dict[str, float]:
    """Ratings after playing `matches` in order from `start` (default: everyone at START)."""
    out = dict(start or {})
    for m in matches:
        update(out, m)
    return out
