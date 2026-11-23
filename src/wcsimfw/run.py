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
- `squad = "sofifa"`: each team's rating is elo + squad_weight x squad rating (`wcsimfw.squad`,
  the roster dated before kickoff). With `squad` on, the fit uses only matches in the year after
  an earlier roster (`squad.windows`), with both teams rated, so a fitted `squad_weight` and
  `squad_weight` fixed at 0 are fitted on the same matches.

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

from wcsimfw import data, elo, match, squad
from wcsimfw.evaluate import evaluate
from wcsimfw.reference import load
from wcsimfw.tournament import STAGES, Tournament

MODELS = {"elo_poisson"}
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
        if p["kind"] == "fitted" and (k not in ("mu", "slope", "squad_weight")
                                      or "fit" not in model):
            raise ValueError(f"fitted param {k!r}: only mu, slope or squad_weight, "
                             "with a [model.fit] table")
    weight = model.get("params", {}).get("squad_weight")
    if (weight is not None) != (model.get("squad") == "sofifa"):
        raise ValueError("squad = 'sofifa' and a squad_weight param go together")
    if weight and weight["kind"] == "fixed" and weight["value"] != 0 and "fit" in model:
        raise ValueError("a fit with a fixed squad_weight other than 0 is not supported")


def _squads(versions, inputs: dict) -> dict[str, dict[str, float]]:
    for v in versions:
        f = f"raw/sofifa/acquisition_2026-10-02/{v}/id-asc/records.jsonl"
        inputs[f] = _sha(data.fetch([f]) / f)
        inputs["data_revision"] = data.REVISION
    return {v: squad.ratings(v) for v in versions}


def strength(cfg: dict, t: Tournament, inputs: dict) -> tuple[dict, dict, dict]:
    """Elo ratings at kickoff, the fitted values (if any) with their standard errors, and the
    squad rating per team at kickoff (empty without `squad`)."""
    model = cfg["model"]
    kickoff = min(m["date"] for m in t.results)
    fitted, sq = {}, {}
    on = model.get("squad") == "sofifa"
    wins = squad.windows(kickoff) if on else []
    if on:
        q = _squads({squad.latest(kickoff)} | {v for v, _, _ in wins}, inputs)
        sq = q[squad.latest(kickoff)]
    if model.get("elo", "reference") == "reference" and "fit" not in model:
        return load("elo", cfg["tournament"])["ratings"], fitted, sq
    rows = elo.history(kickoff)
    inputs["raw/international_results/results.csv"] = _sha(
        data.fetch(["raw/international_results/*"]) / "raw/international_results/results.csv")
    inputs["data_revision"] = data.REVISION
    ratings, sample = {}, []
    keep = FIT[model.get("fit", {}).get("matches", "finals")]
    since = model.get("fit", {}).get("since", "")
    for m in rows:
        gap = elo.update(ratings, m)
        if m["date"] < since or not keep(m):
            continue
        if not on:
            sample.append((gap, not m["neutral"], *m["goals"], 0.0))
            continue
        for v, start, end in wins:
            r = q[v]
            if start <= m["date"] < end and m["home"] in r and m["away"] in r:
                sample.append((gap, not m["neutral"], *m["goals"], r[m["home"]] - r[m["away"]]))
    if "fit" in model:
        x = torch.tensor(sample, dtype=torch.float64).T
        fit_squad = on and model["params"]["squad_weight"]["kind"] == "fitted"
        fitted = match.fit_elo_poisson(*x[:4], squad_gap=x[4] if fit_squad else None)
        fitted["matches"] = (len(sample), 0.0)
    if model.get("elo", "reference") == "reference":
        ratings = load("elo", cfg["tournament"])["ratings"]
    return ratings, fitted, sq


def run(config: Path, out: Path = Path("runs")) -> Path:
    cfg = tomllib.loads(config.read_text())
    check(cfg)
    dev = torch.device(cfg.get("device", "cpu"))
    t = Tournament.load(cfg["tournament"])
    ref = files("wcsimfw.reference").joinpath(cfg["tournament"])
    inputs = {f"reference/{cfg['tournament']}/{n}.json": _sha(ref / f"{n}.json")
              for n in ("format", "teams", "results", "elo")}
    start, fitted, sq = strength(cfg, t, inputs)
    params = {k: fitted[k][0] if p["kind"] == "fitted" else p["value"]
              for k, p in cfg["model"].get("params", {}).items()}
    weight = params.pop("squad_weight", 0.0)
    bonus = {x: weight * sq[x] for x in t.teams} if sq else {}

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
        rating = torch.tensor([r[x] + bonus.get(x, 0.0) for x in t.teams],
                              dtype=torch.float64, device=dev)
        full, adv = match.elo_pairs(rating, **params)
        e = evaluate(t, full, adv, after, cfg["draws"], cfg["seed"], cfg.get("chunk", 50_000))
        seconds[origin] = round(time.perf_counter() - clock, 2)
        tables["scores"].append({"origin": origin} | e["scores"])
        s = {k: v.cpu() for k, v in e["stages"].items()}
        tables["stages"] += [{"origin": origin, "team": team, "elo": r[team],
                              "squad": sq.get(team), "stage": stage,
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
    meta = {"git": _git(), "inputs": inputs, "params": params | {"squad_weight": weight},
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
        print(json.loads((folder / "meta.json").read_text())["fitted"] or "")
        print(folder)


if __name__ == "__main__":
    main()
