"""Run: one TOML file says which model to play, from which origins, and how many times.

Football sentence: write the team sheet down before the match, so anyone can replay the same
forecast and see which numbers were measured and which were chosen.
Pipeline place: the whole pipeline (strength -> match -> tournament -> evaluation), driven by a
config. Every model parameter in the config is marked `fitted` or `fixed` with its `source`.

Usage: `python -m wcsimfw.run configs/elo_poisson.toml [out_dir]` (default out_dir `runs`).
Writes `<out_dir>/<config name>-<time>/`: the config, `meta.json` (git commit, data revision,
versions, device, seconds per origin) and Parquet tables `scores`, `stages`, `matches`,
`calibration`, one row set per origin.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import tomllib
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import polars as pl
import torch

from wcsimfw import data, match
from wcsimfw.evaluate import evaluate
from wcsimfw.reference import load
from wcsimfw.tournament import STAGES, Tournament

MODELS = {"elo_poisson"}


def _git() -> str | None:
    r = subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True,
                       text=True, cwd=Path(__file__).parent)
    return r.stdout.strip() or None


def check(cfg: dict) -> None:
    """Fail early on a config the run cannot honour."""
    if cfg["model"]["name"] not in MODELS:
        raise ValueError(f"unknown model {cfg['model']['name']!r}; known: {sorted(MODELS)}")
    for k, p in cfg["model"].get("params", {}).items():
        if p.get("kind") not in ("fitted", "fixed") or not p.get("source") or "value" not in p:
            raise ValueError(f"param {k!r} needs value, kind (fitted|fixed) and source")


def run(config: Path, out: Path = Path("runs")) -> Path:
    cfg = tomllib.loads(config.read_text())
    check(cfg)
    dev = torch.device(cfg.get("device", "cpu"))
    t = Tournament.load(cfg["tournament"])
    ratings = load("elo", cfg["tournament"])["ratings"]
    elo = torch.tensor([ratings[x] for x in t.teams], dtype=torch.float64, device=dev)
    full, adv = match.elo_pairs(elo, **{k: p["value"] for k, p in cfg["model"]["params"].items()})

    folder = out / f"{config.stem}-{datetime.now():%Y%m%d-%H%M%S}"
    folder.mkdir(parents=True)
    shutil.copy(config, folder / "config.toml")
    tables = {k: [] for k in ("scores", "stages", "matches", "calibration")}
    seconds = {}
    for origin in cfg["origins"]:
        start = time.perf_counter()
        e = evaluate(t, full, adv, None if origin == "kickoff" else origin,
                     cfg["draws"], cfg["seed"], cfg.get("chunk", 50_000))
        seconds[origin] = round(time.perf_counter() - start, 2)
        tables["scores"].append({"origin": origin} | e["scores"])
        s = {k: v.cpu() for k, v in e["stages"].items()}
        tables["stages"] += [{"origin": origin, "team": team, "stage": stage,
                              **{k: s[k][i, j].item() for k in s}}
                             for i, team in enumerate(t.teams) for j, stage in enumerate(STAGES)]
        tables["matches"] += [{"origin": origin, "stage": m["stage"], "home": t.teams[m["home"]],
                               "away": t.teams[m["away"]], "result": m["result"],
                               "p_home": m["p"][0], "p_draw": m["p"][1], "p_away": m["p"][2]}
                              for m in e["matches"]]
        tables["calibration"] += [{"origin": origin, "kind": kind, "bin": b, "n": int(r[0]),
                                   "forecast": r[1], "observed": r[2]}
                                  for kind, c in e["calibration"].items()
                                  for b, r in enumerate(c.tolist())]
    for name, rows in tables.items():
        pl.DataFrame(rows).write_parquet(folder / f"{name}.parquet")
    meta = {"git": _git(), "data_revision": data.REVISION, "wcsimfw": version("wcsimfw"),
            "torch": torch.__version__, "device": str(dev), "seconds": seconds}
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    return folder


def main() -> None:
    args = sys.argv[1:]
    if not args:
        sys.exit("usage: python -m wcsimfw.run CONFIG.toml [OUT_DIR]")
    folder = run(Path(args[0]), Path(args[1]) if len(args) > 1 else Path("runs"))
    with pl.Config(tbl_cols=-1, tbl_width_chars=200):
        print(pl.read_parquet(folder / "scores.parquet"))
    print(folder)


if __name__ == "__main__":
    main()
