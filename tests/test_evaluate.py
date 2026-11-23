import json

import pytest
import torch

from wcsimfw import evaluate as ev
from wcsimfw import match, run, tournament
from wcsimfw.reference import load

T = tournament.Tournament.load()
ELO = torch.tensor([load("elo")["ratings"][t] for t in T.teams], dtype=torch.float64)
FULL, ADV = match.elo_pairs(ELO)


def test_reached_counts():
    assert ev.reached(T).sum(0).tolist() == [32, 16, 8, 4, 2, 1]


def test_targets_per_origin():
    counts = [len(ev.targets(T, a)) for a in (None, *tournament.ORDER[:-1])]
    assert counts == [72, 48, 24, 16, 8, 4, 2, 1]


def test_match_scores_known_values():
    p = torch.tensor([[1.0, 0, 0], [1 / 3, 1 / 3, 1 / 3], [0.5, 0.3, 0.2]])
    s = ev.match_scores(p, torch.tensor([0, 0, 2]))
    assert s["match_rps"].tolist() == pytest.approx([0, 5 / 18, (0.25 + 0.64) / 2])
    assert s["brier"].tolist() == pytest.approx([0, 2 / 3, 0.25 + 0.09 + 0.64])
    assert s["log_score"].tolist() == pytest.approx([0, 1.0986123, 1.6094379])


def test_calibration_bins():
    c = ev.calibration(torch.tensor([0.05, 0.15, 0.95, 1.0]), torch.tensor([0, 1, 1, 0]))
    assert c[0].tolist() == pytest.approx([1, 0.05, 0, -0.05 / 0.0475**0.5])
    assert c[9, :3].tolist() == pytest.approx([2, 0.975, 0.5])
    assert c[:, 0].sum() == 4


def test_paired():
    assert ev.paired(torch.tensor([0.2, 0.3]), torch.tensor([0.1, 0.3])) == pytest.approx(
        (-0.05, 0.05))


def test_perfect_forecast_scores_zero():
    e = ev.evaluate(T, FULL, ADV, after="sf", n=1000)
    p = e["stages"]["exact"]
    assert ev.stage_rps(ev.reached(T).to(p), ev.reached(T).to(p)).sum() == 0
    assert 0 < e["scores"]["stage_rps"] < 0.01  # only the final is left open


def test_scores_better_than_uniform_at_kickoff():
    s = ev.evaluate(T, FULL, ADV, n=2000)["scores"]
    assert s["matches"] == 72 and s["match_rps"] < s["match_rps_uniform"]


def test_run_writes_tables(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text("""
tournament = "wc2026"
origins = ["kickoff", "qf"]
draws = 500
seed = 1
[model]
name = "elo_poisson"
[model.params]
mu = { value = 1.3, kind = "fixed", source = "test" }
""")
    out = run.run(cfg, tmp_path / "runs")
    assert {p.name for p in out.iterdir()} >= {"config.toml", "meta.json", "scores.parquet",
                                               "stages.parquet", "matches.parquet",
                                               "calibration.parquet"}
    meta = json.loads((out / "meta.json").read_text())
    assert set(meta["seconds"]) == {"kickoff", "qf"} and "data_revision" not in meta["inputs"]
    assert run.compare(out, out)["d_stage_rps"].to_list() == [0, 0]


def test_run_rejects_bad_params():
    with pytest.raises(ValueError, match="kind"):
        run.check({"model": {"name": "elo_poisson", "params": {"mu": {"value": 1.3}}}})
    with pytest.raises(ValueError, match="fit"):
        run.check({"model": {"name": "elo_poisson",
                             "params": {"mu": {"kind": "fitted", "source": "x"}}}})
