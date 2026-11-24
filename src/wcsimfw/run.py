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
- A `fitted` parameter (mu, slope) is measured by `match.fit_elo_poisson` on history matches
  before kickoff chosen by `[model.fit]`: `since` (date) and `matches` ("finals": World Cup and
  continental finals; "competitive": no friendlies; "all"). The Elo gaps of the fit come from
  the history, so fit with `elo = "history"` to keep one rating scale.
- `[model.players]` (settings of `players.team_features`, e.g. `source = "sofifa"`): team
  features from each registered squad (processed `players` tables). mu, slope and the feature
  weights (param `features`) are fitted together on the matches of the earlier tournaments in
  those tables, with Elo from the history; the features give each team an attack and a defence.

Model `possession_chain` (`wcsimfw.chain`): goal rates from a possession Markov chain in the
Elo gap; the chain (param `chain`, fitted) is measured on the moves of the earlier tournaments
in the processed `possessions` tables, with Elo from the history (`elo = "history"`).

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
import shutil
import subprocess
import sys
import time
import tomllib
from datetime import datetime
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

MODELS = {"elo_poisson", "possession_chain"}
FIT = {"finals": lambda m: elo.k_factor(m["competition"]) >= 50,
       "competitive": lambda m: m["competition"] != "Friendly", "all": lambda m: True}


def _git() -> str | None:
    r = subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True,
                       text=True, cwd=Path(__file__).parent)
    return r.stdout.strip() or None


def _sha(path) -> str:
    return hashlib.sha256(Path(str(path)).read_bytes()).hexdigest()


def check(cfg: dict) -> None:
    """Fail early on a config the run cannot honour."""
    model = cfg["model"]
    if model["name"] not in MODELS:
        raise ValueError(f"unknown model {model['name']!r}; known: {sorted(MODELS)}")
    if model.get("elo", "reference") not in ("reference", "history"):
        raise ValueError("model.elo must be 'reference' or 'history'")
    for k, p in model.get("params", {}).items():
        if p.get("kind") not in ("fitted", "fixed") or not p.get("source"):
            raise ValueError(f"param {k!r} needs kind (fitted|fixed) and source")
        if p["kind"] == "fixed" and "value" not in p:
            raise ValueError(f"fixed param {k!r} needs a value")
        if p["kind"] == "fitted" and k not in ("mu", "slope", "features", "chain"):
            raise ValueError(f"fitted param {k!r}: only mu, slope, features or chain")
    params = model.get("params", {})
    if model["name"] == "possession_chain":
        if (model.get("elo") != "history" or "fit" in model or "players" in model
                or params.get("chain", {}).get("kind") != "fitted"
                or set(params) - {"chain", "extra_time", "shootout"}):
            raise ValueError("possession_chain: elo = 'history', a fitted `chain` param, "
                             "only extra_time and shootout besides")
        return
    if "chain" in params:
        raise ValueError("the `chain` param belongs to model possession_chain")
    fitted = any(p["kind"] == "fitted" for p in params.values())
    if "players" in model:
        if "fit" in model or any(params.get(k, {}).get("kind") != "fitted"
                                 for k in ("mu", "slope", "features")):
            raise ValueError("[model.players] fits mu, slope and features together "
                             "(no [model.fit])")
    elif "features" in params or (fitted and "fit" not in model):
        raise ValueError("fitted mu or slope need [model.fit]; features need [model.players]")


def _players(cfg: dict, kickoff: str, inputs: dict) -> tuple[dict, tuple[dict, dict]]:
    """Fit on earlier tournaments; attack and defence (log-rate offsets) per team now."""
    tab = {n: pipeline.load(n) for n in ("players", "form", "matches")}
    inputs["processed/players/manifest.json"] = _sha(
        data.data_dir() / "processed/players/manifest.json")
    feats = P.team_features(tab["players"], tab["form"], **cfg["model"]["players"])
    train = P.with_elo(tab["matches"].filter(pl.col("date") < kickoff))
    kept, x, names = P.design(train, feats)
    fitted = match.fit_elo_poisson(*x.values(), names=names)
    fitted["matches"] = (kept.height, 0.0)
    now = {r["team"]: r for r in feats.filter(pl.col("tournament") == cfg["tournament"]).to_dicts()}

    def offset(cols):
        return {team: sum(fitted[c][0] * r[c] for c in cols) for team, r in now.items()}

    att = offset([c for c in names if not c.startswith("def_")])
    dfn = offset([c for c in names if c.startswith("def_")]) if "def_q" in names else att
    return fitted, (att, dfn)


def _chain(kickoff: str, inputs: dict) -> tuple[dict, dict]:
    """Fit the possession chain on the tournaments before kickoff."""
    m = P.with_elo(possessions.load("matches").filter(pl.col("date") < kickoff))
    inputs["processed/possessions/manifest.json"] = _sha(
        data.data_dir() / "processed/possessions/manifest.json")
    ids = m["match_id"].implode()
    moves = possessions.load("moves").filter(pl.col("match_id").is_in(ids))
    x = chain.counts(moves, possessions.load("starts").filter(pl.col("match_id").is_in(ids)),
                     chain.team_gaps(m))
    theta, cov = chain.fit(*x)
    n, own = chain.constants(m, moves)
    rate = [n * own * chain.goal_per_possession(theta, torch.tensor(g)).item() for g in (0., 200.)]
    fitted = {"matches": (m.height, 0.0), "possessions": (n, 0.0), "own_goals": (own, 0.0),
              "goals_at_gap_0": (rate[0], float("nan")),
              "goals_ratio_per_200": (rate[1] / rate[0], float("nan")),
              "chain": (theta.tolist(), cov.diag().sqrt().tolist())}
    return fitted, {"theta": theta, "n": n, "own_goals": own}


def strength(cfg: dict, t: Tournament, inputs: dict) -> tuple[dict, dict, tuple[dict, dict]]:
    """Elo ratings at kickoff, the fitted values (if any) with their standard errors, and the
    attack and defence offsets per team (empty without `[model.players]`)."""
    model = cfg["model"]
    kickoff = min(m["date"] for m in t.results)
    fitted, off = {}, ({}, {})
    if "players" in model:
        fitted, off = _players(cfg, kickoff, inputs)
    if model.get("elo", "reference") == "reference" and "fit" not in model:
        return load("elo", cfg["tournament"])["ratings"], fitted, off
    rows = elo.history(kickoff)
    inputs["raw/international_results/results.csv"] = _sha(
        data.fetch(["raw/international_results/*"]) / "raw/international_results/results.csv")
    inputs["data_revision"] = data.REVISION
    ratings, sample = {}, []
    keep = FIT[model.get("fit", {}).get("matches", "finals")]
    since = model.get("fit", {}).get("since", "")
    for m in rows:
        gap = elo.update(ratings, m)
        if m["date"] >= since and keep(m):
            sample.append((gap, not m["neutral"], *m["goals"]))
    if "fit" in model:
        fitted = match.fit_elo_poisson(*torch.tensor(sample, dtype=torch.float64).T)
        fitted["matches"] = (len(sample), 0.0)
    if model.get("elo", "reference") == "reference":
        ratings = load("elo", cfg["tournament"])["ratings"]
    return ratings, fitted, off


def run(config: Path, out: Path = Path("runs")) -> Path:
    cfg = tomllib.loads(config.read_text())
    check(cfg)
    dev = torch.device(cfg.get("device", "cpu"))
    t = Tournament.load(cfg["tournament"])
    ref = files("wcsimfw.reference").joinpath(cfg["tournament"])
    inputs = {f"reference/{cfg['tournament']}/{n}.json": _sha(ref / f"{n}.json")
              for n in ("format", "teams", "results", "elo")}
    start, fitted, (att, dfn) = strength(cfg, t, inputs)
    fit_chain = {}
    if cfg["model"]["name"] == "possession_chain":
        more, fit_chain = _chain(min(m["date"] for m in t.results), inputs)
        fitted |= more
    params = {k: fitted[k][0] if p["kind"] == "fitted" else p["value"]
              for k, p in cfg["model"].get("params", {}).items() if k not in ("features", "chain")}
    offsets = {}
    if att:
        missing = [x for x in t.teams if x not in att]
        if missing:
            raise ValueError(f"no player features for {missing}")
        offsets = {k: torch.tensor([v[x] for x in t.teams], dtype=torch.float64, device=dev)
                   for k, v in (("attack", att), ("defence", dfn))}

    folder = out / f"{config.stem}-{datetime.now():%Y%m%d-%H%M%S}"
    folder.mkdir(parents=True)
    shutil.copy(config, folder / "config.toml")
    tables = {k: [] for k in ("scores", "stages", "matches", "calibration")}
    seconds = {}
    for origin in cfg["origins"]:
        clock = time.perf_counter()
        after = None if origin == "kickoff" else origin
        r = (elo.ratings(elo.tournament_matches(t.played(after)), start)
             if cfg["model"].get("update") else start)
        rating = torch.tensor([r[x] for x in t.teams], dtype=torch.float64, device=dev)
        full, adv = (chain.pairs(rating, **fit_chain, **params) if fit_chain
                     else match.elo_pairs(rating, **params, **offsets))
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
            "seconds": seconds}
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
