"""Run: one TOML file says which model to play, from which origins, and how many times.

Football sentence: write the team sheet down before the match, so anyone can replay the same
forecast and see which numbers were measured and which were chosen.
Pipeline place: the whole pipeline (strength -> match -> tournament -> evaluation), driven by a
config. Every model parameter is marked `fitted` or `fixed` with its `source`.

Model `elo_poisson` (see `configs/`):
- `elo = "reference"`: the dated Elo table shipped with the tournament (`elo.json`);
  `elo = "history"`: Elo played forward over the international results before kickoff (data).
- `update = true`: before each origin, the ratings also play the tournament matches already
  played (K = 60), so the model knows what happened in the tournament so far.
- Params `mu`, `slope`, `home` (bonus of a host nation in its own country), `level` (how many
  goals history counts for, `match.level`: before each origin the goals of the matches already
  played move the goal level of the tournament). `home` and `level` are off when not named.
- A `fitted` parameter is measured on history matches before kickoff chosen by `[model.fit]`:
  `since` (date) and `matches` ("finals": World Cup and continental finals; "competitive": no
  friendlies; "all"). The Elo gaps of the fit come from the history, so fit with
  `elo = "history"` to keep one rating scale.
- `[model.players]` (settings of `players.team_features`, e.g. `source = "sofifa"`): what each
  registered squad says beyond its rating (`players.residuals`, processed `players` tables).
  The weights (param `features`, fitted) are fitted together with mu and slope on the `[model.fit]`
  matches, where sides without a rated squad count as their rating says.

Model `possession_chain` (`wcsimfw.chain`): goal rates from a possession Markov chain in the Elo
gap; `chain` (and `home`, `level`) are fitted on the earlier tournaments in the processed
`possessions` tables, with Elo from the history (`elo = "history"`). `style` (fixed, in matches):
each side's own moves in those tournaments and in the matches played before the origin.

Usage:
  python -m wcsimfw.run CONFIG.toml [OUT_DIR]   (default OUT_DIR `runs`)
  python -m wcsimfw.run compare RUN_A RUN_B     (B - A per origin, paired; negative = B better)
A run writes `<OUT_DIR>/<config name>-<time>/`: the config, `meta.json` (git commit, sha256 of
every input file, data revision if the data were read, fitted values, versions, device,
seconds per origin) and Parquet tables `scores`, `stages`, `matches`, `calibration`.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
import tomllib
from datetime import date, datetime
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path

import polars as pl
import torch

from wcsimfw import chain, data, elo, match
from wcsimfw import players as P
from wcsimfw.evaluate import evaluate
from wcsimfw.pipelines import players as pipeline
from wcsimfw.pipelines import possessions
from wcsimfw.reference import load
from wcsimfw.tournament import STAGES, Tournament

FIT = {"finals": lambda m: elo.k_factor(m["competition"]) >= 50,
       "competitive": lambda m: m["competition"] != "Friendly", "all": lambda m: True}


def _git() -> str | None:
    r = subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True,
                       text=True, cwd=Path(__file__).parent)
    return r.stdout.strip() or None


def _sha(path) -> str:
    return hashlib.sha256(Path(str(path)).read_bytes()).hexdigest()


FITTED = {"elo_poisson": {"mu", "slope", "home", "level", "features"},
          "possession_chain": {"chain", "home", "level"}}
PARAMS = {k: v | {"extra_time", "shootout"} | ({"style"} if k == "possession_chain" else set())
          for k, v in FITTED.items()}


def check(cfg: dict) -> None:
    """Fail early on a config the run cannot honour."""
    model = cfg["model"]
    name, params = model["name"], model.get("params", {})
    if name not in FITTED:
        raise ValueError(f"unknown model {name!r}; known: {sorted(FITTED)}")
    if model.get("elo", "reference") not in ("reference", "history"):
        raise ValueError("model.elo must be 'reference' or 'history'")
    for k, p in params.items():
        if k not in PARAMS[name]:
            raise ValueError(f"{name} has no param {k!r}; known: {sorted(PARAMS[name])}")
        if p.get("kind") not in ("fitted", "fixed") or not p.get("source"):
            raise ValueError(f"param {k!r} needs kind (fitted|fixed) and source")
        if p["kind"] == "fixed" and "value" not in p:
            raise ValueError(f"fixed param {k!r} needs a value")
        if p["kind"] == "fitted" and k not in FITTED[name]:
            raise ValueError(f"param {k!r} cannot be fitted")
    fitted = {k for k, p in params.items() if p["kind"] == "fitted"}
    if name == "possession_chain":
        if (model.get("elo") != "history" or "fit" in model or "players" in model
                or "chain" not in fitted):
            raise ValueError("possession_chain: elo = 'history', a fitted `chain` param, "
                             "no [model.fit] and no [model.players]")
        return
    if fitted and "fit" not in model:
        raise ValueError("fitted params need [model.fit]")
    if ("players" in model) != ("features" in fitted) or (
            "players" in model and not {"mu", "slope"} <= fitted):
        raise ValueError("[model.players] needs mu, slope and features fitted together, and "
                         "features need [model.players]")


def _level(rows: list[tuple[str, str, float, float]]) -> float:
    """`match.fit_level` of (competition, date, goals, expected goals) rows in date order: one
    tournament is a run of matches of one competition without a break of 120 days."""
    total, last = {}, {}
    for comp, day, goals, expected in rows:
        d = date.fromisoformat(day)
        if comp not in last or (d - last[comp][1]).days > 120:
            last[comp] = (day, d)
        last[comp] = (last[comp][0], d)
        t = total.setdefault((comp, last[comp][0]), [0.0, 0.0])
        t[0], t[1] = t[0] + goals, t[1] + expected
    return match.fit_level(*torch.tensor(list(total.values()), dtype=torch.float64).T)


def _squads(cfg: dict, sample: list[dict], rows: list[dict], inputs: dict):
    """Squad features beyond the rating: their values in the fit sample (x_home, x_away), their
    names, and per team now."""
    tab = {n: pipeline.load(n) for n in ("players", "form", "matches")}
    inputs[f"processed/{pipeline.ARTIFACT}/manifest.json"] = _sha(
        data.processed(pipeline.ARTIFACT, pipeline.build) / "manifest.json")
    kick = dict(tab["players"].select("tournament", "kickoff").unique().iter_rows())
    now = cfg["tournament"]
    earlier = [t for t in kick if kick[t] < kick[now]]
    feats = P.team_features(tab["players"], tab["form"], **cfg["model"]["players"]).filter(
        pl.col("tournament").is_in([*earlier, now]))
    feats = P.residuals(feats, {t: kick[t] for t in [*earlier, now]}, rows, earlier)
    x_home, x_away, names, _ = P.covariates(
        sample, feats, tab["matches"].filter(pl.col("tournament").is_in(earlier)))
    team = {r["team"]: r for r in feats.filter(pl.col("tournament") == now).to_dicts()}
    return x_home, x_away, names, team


def elo_model(cfg: dict, t: Tournament, inputs: dict) -> tuple[dict, dict, tuple[dict, dict]]:
    """Elo ratings at kickoff, the fitted values with their standard errors, and the attack and
    defence offsets per team (empty without `[model.players]`)."""
    model, params = cfg["model"], cfg["model"].get("params", {})
    reference = model.get("elo", "reference") == "reference"
    ratings = load("elo", cfg["tournament"])["ratings"] if reference else {}
    if reference and "fit" not in model:
        return ratings, {}, ({}, {})
    rows = elo.history(min(m["date"] for m in t.results))
    inputs["raw/international_results/results.csv"] = _sha(
        data.fetch(["raw/international_results/*"]) / "raw/international_results/results.csv")
    inputs["data_revision"] = data.REVISION
    now, sample = {}, []
    keep = FIT[model.get("fit", {}).get("matches", "finals")]
    since = model.get("fit", {}).get("since", "")
    for m in rows:
        gap = elo.update(now, m)
        if m["date"] >= since and keep(m):
            sample.append(m | {"gap": gap})
    ratings = ratings or now
    if "fit" not in model:
        return ratings, {}, ({}, {})
    x = torch.tensor([[m["gap"], not m["neutral"], *m["goals"]] for m in sample],
                     dtype=torch.float64).T
    xh = xa = None
    names, team = [], {}
    if "players" in model:
        xh, xa, names, team = _squads(cfg, sample, rows, inputs)
    fitted = match.fit_elo_poisson(*x, xh, xa, names) | {"matches": (len(sample), 0.0)}
    if "level" in params:
        theta, _ = match.fit_poisson(*x, xh, xa)
        e = match.expected_goals(theta, x[0], x[1], xh, xa).sum(-1).tolist()
        fitted["level"] = (_level([(m["competition"], m["date"], sum(m["goals"]), v)
                                   for m, v in zip(sample, e, strict=True)]), float("nan"))
    if names:
        fitted["squad_matches"] = (int((xh != 0).any(-1).sum()), 0.0)

    def offset(cols):
        return {x: sum(fitted[c][0] * team[x][c] for c in cols) if x in team else 0.0
                for x in t.teams} if names else {}

    att = offset([c for c in names if not c.startswith("def_")])
    dfn = offset([c for c in names if c.startswith("def_")]) if any(
        c.startswith("def_") for c in names) else att
    return ratings, fitted, (att, dfn)


class Chain:
    """The possession chain fitted on the tournaments before kickoff, as goal rates per origin."""

    def __init__(self, cfg: dict, t: Tournament, inputs: dict):
        self.t, self.params = t, cfg["model"]["params"]
        kickoff = min(m["date"] for m in t.results)
        inputs[f"processed/{possessions.ARTIFACT}/manifest.json"] = _sha(
            data.processed(possessions.ARTIFACT, possessions.build) / "manifest.json")
        self.tab = {k: possessions.load(k) for k in ("matches", "moves", "starts")}
        self.all = P.with_elo(self.tab["matches"])
        self.old = m = self.all.filter(pl.col("date") < kickoff)
        ids = m["match_id"].implode()
        x = chain.counts(*(self.tab[k].filter(pl.col("match_id").is_in(ids))
                           for k in ("moves", "starts")), chain.team_gaps(m))
        self.theta, cov = chain.fit(*x)
        self.n, self.own = chain.constants(m, self.tab["moves"])
        r = chain.residuals(self.theta, m, self.n * self.own)
        home = r.filter("at_home")
        y, e = home["goals_home"].sum(), home["e_home"].sum()
        rate = [self.n * self.own * chain.goal_per_possession(self.theta, torch.tensor(g)).item()
                for g in (0.0, 200.0)]
        levels = r.sort("date").select("tournament", "date", pl.col("goals_home") + pl.col(
            "goals_away"), pl.col("e_home") + pl.col("e_away")).rows()
        nan = float("nan")
        self.fitted = {
            "matches": (m.height, 0.0), "possessions": (self.n, 0.0), "own_goals": (self.own, 0.0),
            "home": (math.log(y / e), y**-0.5), "level": (_level(levels), nan),
            "goals_at_gap_0": (rate[0], nan), "goals_ratio_per_200": (rate[1] / rate[0], nan),
            "chain": (self.theta.tolist(), cov.diag().sqrt().tolist())}

    def rates(self, rating: torch.Tensor, after: str | None) -> torch.Tensor:
        style = None
        if "style" in self.params:
            pairs = {frozenset((m["home"], m["away"])) for m in self.t.played(after)}
            now = self.all.filter(pl.col("date") > self.old["date"].max()).filter(
                pl.struct("home", "away").map_elements(
                    lambda r: frozenset(r.values()) in pairs, pl.Boolean))
            style = chain.style(self.theta, self.tab["moves"], self.tab["starts"],
                                pl.concat([self.old, now]), self.t.teams,
                                self.params["style"]["value"])
        return chain.rates(rating, self.theta, self.n, self.own, style)


def goal_level(t: Tournament, rate: torch.Tensor, after: str | None, prior: float) -> float:
    """`match.level` from the matches played before the origin; `rate` [places, T, T]."""
    idx = {x: i for i, x in enumerate(t.teams)}
    y = e = 0.0
    for m in t.played(after):
        c, h, a = 1 + t.hosts.index(m["country"]), idx[m["home"]], idx[m["away"]]
        y, e = y + sum(m["goals_90"]), e + (rate[c, h, a] + rate[c, a, h]).item()
    return match.level(y, e, prior)


def run(config: Path, out: Path = Path("runs")) -> Path:
    cfg = tomllib.loads(config.read_text())
    check(cfg)
    dev = torch.device(cfg.get("device", "cpu"))
    t = Tournament.load(cfg["tournament"])
    ref = files("wcsimfw.reference").joinpath(cfg["tournament"])
    inputs = {f"reference/{cfg['tournament']}/{n}.json": _sha(ref / f"{n}.json")
              for n in ("format", "teams", "results", "elo")}
    start, fitted, (att, dfn) = elo_model(cfg, t, inputs)
    possession = None
    if cfg["model"]["name"] == "possession_chain":
        possession = Chain(cfg, t, inputs)
        fitted |= possession.fitted
    params = {k: fitted[k][0] if p["kind"] == "fitted" else p["value"]
              for k, p in cfg["model"].get("params", {}).items() if k not in ("features", "chain")}
    tie = {k: params[k] for k in ("extra_time", "shootout") if k in params}
    number = {k: params[k] for k in ("mu", "slope") if k in params}
    offsets = {k: torch.tensor([v[x] for x in t.teams], dtype=torch.float64, device=dev)
               for k, v in (("attack", att), ("defence", dfn)) if v}
    host = t.host_bonus(params.get("home", 0.0)).to(dev)

    folder = out / f"{config.stem}-{datetime.now():%Y%m%d-%H%M%S}"
    folder.mkdir(parents=True)
    shutil.copy(config, folder / "config.toml")
    tables = {k: [] for k in ("scores", "stages", "matches", "calibration")}
    seconds, level = {}, {}
    for origin in cfg["origins"]:
        clock = time.perf_counter()
        after = None if origin == "kickoff" else origin
        r = (elo.ratings(elo.tournament_matches(t.played(after)), start)
             if cfg["model"].get("update") else start)
        rating = torch.tensor([r[x] for x in t.teams], dtype=torch.float64, device=dev)
        rate = (possession.rates(rating, after) if possession
                else match.pair_rates(rating, **number, **offsets))
        if "level" in params:
            level[origin] = goal_level(t, rate * torch.exp(host)[:, :, None], after,
                                       params["level"])
            rate = rate * level[origin]
        full, adv = match.pairs(rate, **tie, host=host)
        e = evaluate(t, full, adv, after, cfg["draws"], cfg["seed"], cfg.get("chunk", 50_000))
        seconds[origin] = round(time.perf_counter() - clock, 2)
        tables["scores"].append({"origin": origin} | e["scores"])
        s = {k: v.cpu() for k, v in e["stages"].items()}
        tables["stages"] += [{"origin": origin, "team": team, "elo": r[team],
                              "attack": att.get(team), "defence": dfn.get(team), "stage": stage,
                              **{k: s[k][i, j].item() for k in s}}
                             for i, team in enumerate(t.teams) for j, stage in enumerate(STAGES)]
        tables["matches"] += [{"origin": origin, "stage": m["stage"], "home": t.teams[m["home"]],
                               "away": t.teams[m["away"]], "result": m["result"],
                               "p_home": m["p"][0], "p_draw": m["p"][1], "p_away": m["p"][2],
                               "match_rps": m["match_rps"], "brier": m["brier"],
                               "log_score": m["log_score"]}
                              for m in e["matches"]]
        tables["calibration"] += [{"origin": origin, "kind": kind, "bin": b, "n": int(row[0]),
                                   "forecast": row[1], "observed": row[2], "z": row[3]}
                                  for kind, c in e["calibration"].items()
                                  for b, row in enumerate(c.tolist())]
    for name, rows in tables.items():
        pl.DataFrame(rows).write_parquet(folder / f"{name}.parquet")
    meta = {"git": _git(), "inputs": inputs, "params": params,
            "fitted": {k: {"value": v, "se": se} for k, (v, se) in fitted.items()},
            "wcsimfw": version("wcsimfw"), "torch": torch.__version__, "device": str(dev),
            "seconds": seconds, "level": level}
    (folder / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return folder


def compare(a: Path, b: Path) -> pl.DataFrame:
    """Score of run b minus run a at each origin, paired over the same matches and teams,
    with standard errors. Negative means b is better."""
    def per_team(x):
        return (pl.read_parquet(x / "stages.parquet")
                .group_by("origin", "team", maintain_order=True)
                .agg(((pl.col("exact") - pl.col("reached")) ** 2).mean().alias("stage_rps")))

    def paired(x, y, on, cols):
        d = x.join(y, on=on, suffix="_b").group_by("origin", maintain_order=True)
        return d.agg([e for c in cols for e in (
            (pl.col(f"{c}_b") - pl.col(c)).mean().alias(f"d_{c}"),
            ((pl.col(f"{c}_b") - pl.col(c)).std() / pl.len().sqrt()).alias(f"d_{c}_se"))])

    on = ["origin", "stage", "home", "away"]
    m = paired(*(pl.read_parquet(x / "matches.parquet") for x in (a, b)), on,
               ["match_rps", "log_score"])
    s = paired(per_team(a), per_team(b), ["origin", "team"], ["stage_rps"])
    return m.join(s, on="origin", how="full", coalesce=True, maintain_order="left")


def main() -> None:
    args = sys.argv[1:]
    with pl.Config(tbl_cols=-1, tbl_width_chars=250, float_precision=5):
        if len(args) == 3 and args[0] == "compare":
            print(compare(Path(args[1]), Path(args[2])))
            return
        if not args:
            sys.exit("usage: python -m wcsimfw.run CONFIG.toml [OUT_DIR] | compare RUN_A RUN_B")
        folder = run(Path(args[0]), Path(args[1]) if len(args) > 1 else Path("runs"))
        print(pl.read_parquet(folder / "scores.parquet").select(
            "origin", "matches", "match_rps", "match_rps_se", "log_score", "stage_rps",
            "stage_rps_se", "stage_rps_mc_se"))
        fitted = json.loads((folder / "meta.json").read_text())["fitted"]
        print({k: v for k, v in fitted.items() if not isinstance(v["value"], list)} or "")
        print(folder)


if __name__ == "__main__":
    main()
