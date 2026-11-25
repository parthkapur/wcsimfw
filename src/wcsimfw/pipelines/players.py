"""Players pipeline: each tournament's registered squads with what was known of every player
before kickoff.

Football sentence: a national side is the 26 players it takes to the tournament, and each of
them arrives with a season of club football behind him.
Pipeline place: data -> processed artifact (input to strength). Nothing is fitted.

Inputs (raw, pinned revision): API-Football squads, player season statistics, pre-tournament club
form and fixtures per tournament (`raw/apifootball/historical/<t>/`; WC2026 squads, players and
form at the top of `raw/apifootball/`), and the dated SoFIFA rosters.
Time rule: the season used is the one labelled the year before the tournament (it ended
before kickoff, also for calendar-year leagues); form matches are dated before kickoff; the SoFIFA
roster is the newest dated before kickoff and at most 400 days older.

Outputs (`$WCSIMFW_DATA/processed/players-v2/`, Parquet + manifest.json):
- `players`: tournament, kickoff, team, player_id, name, birth, position, minutes, rating
  (minutes-weighted season match rating; null for tournaments without player statistics),
  top_minutes (minutes in the five big leagues and the
  Champions League), sofifa (overall, joined on nationality, age and name; null if not found).
- `form`: tournament, player_id, date, minutes, rating, league_id (club matches before kickoff).
- `matches`: tournament, date, home, away, goals after 90 minutes (final tournament only).
"""

from __future__ import annotations

import csv
import hashlib
import json
import unicodedata
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

import polars as pl

from wcsimfw import data, elo

ARTIFACT = "players-v2"  # new version when the tables change
# SoFIFA roster -> date the vendor shows for it.
SOFIFA = {"eafc/fifa10.csv": "2010-02-22", "eafc/fifa14.csv": "2014-06-06",
          **{f"sofifa/acquisition_2026-10-02/{v}/id-asc/records.jsonl": d for v, d in
             (("180076", "2018-06-11"), ("230006", "2022-11-16"), ("240043", "2024-06-12"),
              ("260037", "2026-06-10"))}}
TOURNAMENTS = ("wc2010", "wc2014", "euro2016", "copa2016", "wc2018", "copa2019", "afcon2019",
               "euro2020", "wc2022", "goldcup2023", "euro2024", "copa2024", "goldcup2025",
               "wc2026")
TOP = {39, 140, 135, 78, 61, 2}  # Premier League, LaLiga, Serie A, Bundesliga, Ligue 1, UCL
# API-Football team names -> reference names (after elo.ALIAS).
ALIAS = {"Bosnia & Herzegovina": "Bosnia and Herzegovina", "Rep. Of Ireland": "Republic of Ireland",
         "FYR Macedonia": "North Macedonia", "Iran": "IR Iran", "USA": "USA",
         "Cape Verde Islands": "Cabo Verde", "Curacao": "Curaçao", "United States": "USA",
         "Turkiye": "Türkiye", "Ireland": "Republic of Ireland",
         "Trinidad & Tobago": "Trinidad and Tobago",
         "St. Kitts and Nevis": "Saint Kitts and Nevis"}


def latest(before: str) -> str | None:
    """The newest SoFIFA roster dated before `before` (YYYY-MM-DD); None if there is none."""
    return max((v for v, d in SOFIFA.items() if d < before), key=SOFIFA.get, default=None)


def overall(text: str) -> int:
    return int(text.split("+")[0].split("-")[0])  # live boosts are shown as "70+2"


def team(name: str) -> str:
    return elo.ALIAS.get(ALIAS.get(name, name), ALIAS.get(name, name))


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _folders(raw: Path, t: str) -> tuple[Path, Path, Path]:
    base = raw / "apifootball" / ("" if t == "wc2026" else f"historical/{t}")
    return base / "squads", base / "players", base / "player_form"


def _fixtures(raw: Path, t: str) -> list[dict]:
    fx = _read(raw / f"apifootball/historical/{t}/fixtures.json")
    done = sorted((f for f in fx if f["score"]["fulltime"]["home"] is not None
                   and f["league"]["round"].lower().find("qualif") < 0),
                  key=lambda f: f["fixture"]["date"])
    days = [date.fromisoformat(f["fixture"]["date"][:10]) for f in done]
    start = max([i for i in range(1, len(days)) if (days[i] - days[i - 1]).days > 60], default=0)
    return done[start:]  # the final tournament: after the last break of 60 days


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return " ".join(s.replace("-", " ").replace(".", " ").split())


def _iso(text: str | None) -> str | None:
    """A vendor date as YYYY-MM-DD ("1980-5-8" too); None if it is not a date."""
    try:
        return date(*map(int, text.split("-"))).isoformat()
    except (AttributeError, TypeError, ValueError):
        return None


def _sofifa(raw: Path, kickoff: str) -> dict[str, list[dict]] | None:
    v = latest(kickoff)
    if v is None or (date.fromisoformat(kickoff) - date.fromisoformat(SOFIFA[v])).days > 400:
        return None
    out: dict[str, list[dict]] = {}
    with open(raw / v, encoding="utf-8") as f:
        rows = list(csv.DictReader(f)) if v.endswith("csv") else [json.loads(x) for x in f]
    for r in rows:
        out.setdefault(team(r["nationality_name"]), []).append(
            {"name": _norm(r["long_name"]), "short": _norm(r["short_name"]),
             "age": int(r["age"]), "overall": overall(r["overall"]),
             "keeper": "GK" in r["player_positions"], "date": SOFIFA[v]})
    return out


def _match_sofifa(p: dict, cands: list[dict]) -> dict | None:
    """The SoFIFA player with the same age (±1) whose name is closest; None if no name is close
    enough (ratio >= 0.6; without a birth date the name alone decides, ratio >= 0.75)."""
    birth = p["birth"]
    names = {_norm(p["name"]), _norm(f"{p['firstname']} {p['lastname']}")}
    best, score = None, 0.6 if birth else 0.75
    for c in cands:
        if birth:
            age = (date.fromisoformat(c["date"]) - date.fromisoformat(birth)).days // 365
            if abs(age - c["age"]) > 1:
                continue
        s = max(SequenceMatcher(None, a, b).ratio() for a in names for b in (c["name"], c["short"]))
        if s > score:
            best, score = c, s
    return best


def build(raw: Path | None = None, out: Path | None = None) -> Path:
    if raw is None:
        pats = [f"raw/apifootball/{'' if t == 'wc2026' else f'historical/{t}/'}{d}/*"
                for t in TOURNAMENTS for d in ("squads", "players", "player_form")]
        pats += [f"raw/apifootball/historical/{t}/fixtures.json" for t in TOURNAMENTS]
        raw = data.fetch(pats + [f"raw/{v}" for v in SOFIFA]
                         + ["raw/international_results/*"]) / "raw"
    out = out or data.data_dir() / "processed" / ARTIFACT
    out.mkdir(parents=True, exist_ok=True)
    players, form, matches = [], [], []
    for t in TOURNAMENTS:
        fx = _fixtures(raw, t)
        kickoff = min(f["fixture"]["date"][:10] for f in fx)
        season = int(kickoff[:4]) - 1
        sq_dir, pl_dir, fm_dir = _folders(raw, t)
        sofifa = _sofifa(raw, kickoff)
        teams = set()
        for f in sorted(sq_dir.glob("*.json")):
            sq = _read(f)
            name = team(sq["team"]["name"])
            teams.add(name)
            for p in sq["players"]:
                path = pl_dir / f"{p['id']}.json"
                prof = _read(path) if path.exists() else []
                prof = [x for x in (prof if isinstance(prof, list) else [prof]) if x]  # {} = none
                info = prof[0]["player"] if prof else {}
                stats = [s for x in prof for s in x.get("statistics", [])
                         if s["league"]["season"] == season and (s["games"]["minutes"] or 0) > 0]
                rated = [s for s in stats if s["games"]["rating"]]
                mins = sum(s["games"]["minutes"] for s in stats)
                rm = sum(s["games"]["minutes"] for s in rated)
                row = {"tournament": t, "kickoff": kickoff, "team": name, "player_id": p["id"],
                       "name": p["name"], "birth": _iso((info.get("birth") or {}).get("date")),
                       "position": p["position"], "minutes": mins,
                       "rating": sum(s["games"]["minutes"] * float(s["games"]["rating"])
                                     for s in rated) / rm if rm else None,
                       "top_minutes": sum(s["games"]["minutes"] for s in stats
                                          if s["league"]["id"] in TOP)}
                hit = _match_sofifa(
                    {"name": p["name"], "firstname": info.get("firstname", ""),
                     "lastname": info.get("lastname", ""), "birth": row["birth"]},
                    sofifa.get(name, [])) if sofifa else None
                row["sofifa"] = hit["overall"] if hit else None
                if hit and row["position"] in (None, "Unknown"):
                    row["position"] = "Goalkeeper" if hit["keeper"] else "Outfield"
                players.append(row)
                fp = fm_dir / f"{p['id']}.json"
                if fp.exists():
                    fd = _read(fp)
                    for m in fd.get("club_matches", fd.get("matches", [])):
                        if m["date"] < kickoff and m.get("rating") and m.get("minutes"):
                            form.append({"tournament": t, "player_id": p["id"], "date": m["date"],
                                         "minutes": m["minutes"], "rating": float(m["rating"]),
                                         "league_id": m["league_id"]})
        for f in fx:
            h, a = team(f["teams"]["home"]["name"]), team(f["teams"]["away"]["name"])
            if h in teams and a in teams:
                g = f["score"]["fulltime"]
                matches.append({"tournament": t, "date": f["fixture"]["date"][:10], "home": h,
                                "away": a, "goals_home": g["home"], "goals_away": g["away"]})
    tables = {"players": players, "form": form, "matches": matches}
    manifest = {"tournaments": TOURNAMENTS, "rows": {k: len(v) for k, v in tables.items()},
                "data_revision": data.REVISION, "files": {}}
    for k, rows in tables.items():
        pl.DataFrame(rows, infer_schema_length=None).write_parquet(out / f"{k}.parquet")
        manifest["files"][k] = hashlib.sha256((out / f"{k}.parquet").read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out


def load(name: str) -> pl.DataFrame:
    """One output table (local, else from the dataset, else built)."""
    return pl.read_parquet(data.processed(ARTIFACT, build) / f"{name}.parquet")


if __name__ == "__main__":
    print(build())
