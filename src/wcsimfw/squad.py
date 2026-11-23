"""Squad rating: how good a nation's best players are, from a dated player-rating snapshot.

Football sentence: a side whose best eleven are better players scores more and concedes less
than its results alone say.
Pipeline place: strength (data -> strength -> match). Input: a SoFIFA roster snapshot (every
player's overall rating and nationality, on one scale across all leagues); output: one number per
nation, the mean overall of its best goalkeeper and its ten best outfield players.
Numbers: nothing is fitted here; 1 goalkeeper + 10 outfield players is the shape of a team.
A run adds the rating to Elo with a fitted weight (Elo points per point of squad rating).

ponytail: the nation's best players in the snapshot, not the registered squad: an injured or
left-out star still counts. Join the official squad list if that turns out to matter.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

from wcsimfw import data
from wcsimfw.elo import ALIAS as _ELO_ALIAS

# Roster version -> date the vendor shows for it (publication time not known more exactly).
VERSIONS = {"180076": "2018-06-11", "230006": "2022-11-16", "240043": "2024-06-12",
            "260037": "2026-06-10"}
ALIAS = _ELO_ALIAS | {"United States": "USA", "Curacao": "Curaçao", "Iran": "IR Iran"}
OUTFIELD = 10


def _overall(text: str) -> int:
    return int(text.split("+")[0].split("-")[0])  # live boosts are shown as "70+2"


def ratings(version: str) -> dict[str, float]:
    """Squad rating per nation (reference names) for one roster version. Nations without a
    goalkeeper and ten outfield players in the snapshot are left out."""
    path = data.fetch([f"raw/sofifa/acquisition_2026-10-02/{version}/id-asc/records.jsonl"])
    players: dict[str, list[tuple[bool, int]]] = {}
    with open(path / f"raw/sofifa/acquisition_2026-10-02/{version}/id-asc/records.jsonl",
              encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            nation = ALIAS.get(r["nationality_name"], r["nationality_name"])
            players.setdefault(nation, []).append((r["best_position"] == "GK",
                                                   _overall(r["overall"])))
    out = {}
    for nation, ps in players.items():
        gk = sorted((o for g, o in ps if g), reverse=True)[:1]
        field = sorted((o for g, o in ps if not g), reverse=True)[:OUTFIELD]
        if len(gk) == 1 and len(field) == OUTFIELD:
            out[nation] = (gk[0] + sum(field)) / (1 + OUTFIELD)
    return out


def latest(before: str) -> str:
    """The newest roster version dated before `before` (YYYY-MM-DD)."""
    return max((v for v, d in VERSIONS.items() if d < before), key=VERSIONS.get)


def windows(before: str, days: int = 365) -> list[tuple[str, str, str]]:
    """(version, start, end): the matches each version can rate, from its date to `days` later,
    the next version or `before`, whichever comes first. Every match is rated by a snapshot
    dated before it."""
    dates = sorted(VERSIONS.items(), key=lambda x: x[1])
    out = []
    for i, (v, d) in enumerate(dates):
        end = min((date.fromisoformat(d) + timedelta(days)).isoformat(), before,
                  *(n for _, n in dates[i + 1:i + 2]))
        if d < end:
            out.append((v, d, end))
    return out
