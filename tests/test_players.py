import polars as pl
import pytest
import torch

from wcsimfw import match, run
from wcsimfw import players as P
from wcsimfw.pipelines import players as pipeline


def squad(team, base, top=0):
    """One keeper and eleven outfield players; the twelfth outfield player has fewest minutes."""
    rows = [{"position": "Goalkeeper", "minutes": 3000, "rating": base, "sofifa": 80}]
    rows += [{"position": "Defender" if i < 5 else "Attacker", "minutes": 2000 - 10 * i,
              "rating": base + (0.5 if i >= 5 else 0), "sofifa": 70 + i} for i in range(11)]
    return [r | {"tournament": "t", "kickoff": "2026-06-11", "team": team, "player_id": i,
                 "top_minutes": top * r["minutes"]} for i, r in enumerate(rows)]


PLAYERS = pl.DataFrame(squad("A", 7.0, 1) + squad("B", 6.5))


def test_eleven_is_keeper_plus_ten_with_most_minutes():
    xi = P.eleven(P.player_values(PLAYERS))
    assert xi.group_by("team").len()["len"].to_list() == [11, 11]
    assert 10 * 2000 - 10 * 10 not in xi["minutes"].to_list()  # the twelfth is left out
    assert P.eleven(P.player_values(PLAYERS, source="sofifa"), "sofifa")["value"].min() == 71


def test_minutes_prior_pulls_toward_position_mean():
    p = PLAYERS.with_columns(pl.when(pl.col("player_id") == 0).then(0).otherwise(
        pl.col("minutes")).alias("minutes"))
    v = P.player_values(p, tau=900).filter(pl.col("player_id") == 0)
    assert v["value"].to_list() == pytest.approx([6.75, 6.75])  # keepers' mean, no own minutes


def test_features_are_centred_and_split():
    f = P.team_features(PLAYERS, top=True).sort("team")
    assert f["q"].sum() == pytest.approx(0) and f["q"][0] > 0
    assert f["top"].to_list() == pytest.approx([0.5, -0.5])
    s = P.team_features(PLAYERS, split=True)
    assert set(s.columns) == {"tournament", "team", "att_q", "def_q"}


def test_covariates_tied_split_and_missing():
    table = pl.DataFrame({"tournament": ["t"], "date": ["2026-06-12"], "home": ["A"],
                          "away": ["B"]})
    rows = [{"date": "2026-06-12", "home": "A", "away": "B"},
            {"date": "2026-06-12", "home": "C", "away": "B"},  # not a tournament match
            {"date": "2020-06-12", "home": "A", "away": "B"}]
    xh, xa, names, tour = P.covariates(rows, P.team_features(PLAYERS), table)
    assert names == ["q"] and tour == ["t", None, None]
    assert xh[0, 0] == -xa[0, 0] > 0 and (xh[1:] == 0).all()
    split = P.team_features(PLAYERS, split=True)
    xh, _, names, _ = P.covariates(rows[:1], split, table)
    assert names == ["att_q", "def_q"]
    # the home rate falls with the away defence
    assert xh[0, 1] == -split.filter(pl.col("team") == "B")["def_q"].item()


def test_side_without_eleven_has_no_features():
    few = PLAYERS.filter((pl.col("team") == "A") | (pl.col("player_id") < 8))
    assert P.team_features(few)["team"].to_list() == ["A"]


def test_residuals_take_out_the_rating():
    f = pl.DataFrame({"tournament": ["t"] * 3, "team": ["A", "B", "C"], "q": [1.0, 0.0, -1.0]})
    h = [{"date": "2020-01-01", "home": "A", "away": "C", "goals": (1, 0),
          "competition": "Friendly", "neutral": True}]
    r = P.residuals(f, {"t": "2021-01-01"}, h, ["t"])  # q is exactly a line in the rating
    assert r["q"].to_list() == pytest.approx([0.0, 0.0, 0.0])
    r = P.residuals(f, {"t": "2019-01-01"}, h, ["t"])  # no match played yet: nothing to take out
    assert r["q"].to_list() == pytest.approx([1.0, 0.0, -1.0])


def test_fit_recovers_feature_weights():
    g = torch.Generator().manual_seed(2)
    n = 20_000
    gap = torch.randn(n, generator=g, dtype=torch.float64) * 300
    xh = torch.randn(n, 2, generator=g, dtype=torch.float64)
    xa = torch.randn(n, 2, generator=g, dtype=torch.float64)
    beta = torch.tensor([0.3, -0.2], dtype=torch.float64)
    lam = torch.tensor(1.2).log()
    gh = torch.poisson((lam + 0.5 * gap / 400 + xh @ beta).exp(), g)
    ga = torch.poisson((lam - 0.5 * gap / 400 + xa @ beta).exp(), g)
    th, cov = match.fit_poisson(gap, torch.zeros(n), gh, ga, xh, xa)
    se = cov.diag().sqrt()
    assert torch.isnan(se[2])  # no home matches: no home term
    for i, true in ((1, 0.5), (3, 0.3), (4, -0.2)):
        assert abs(th[i] - true) < 4 * se[i]


def test_attack_and_defence_offsets():
    elo = torch.tensor([1800.0, 1800.0], dtype=torch.float64)
    full, _ = match.elo_pairs(elo, attack=torch.tensor([0.0, 0.0]), defence=torch.tensor(
        [0.0, torch.log(torch.tensor(2.0))]))
    goals = torch.arange(11, dtype=torch.float64)
    home_goals = (full[0, 1].sum(-1) * goals).sum()
    assert home_goals == pytest.approx(match.MU / 2, rel=1e-3)  # a defence that halves the rate


def test_sofifa_join_by_age_and_name():
    cands = [{"name": "lionel andres messi cuccittini", "short": "l messi", "age": 38,
              "overall": 86, "keeper": False, "date": "2026-06-10"},
             {"name": "lionel messi", "short": "l messi", "age": 20, "overall": 60,
              "keeper": False, "date": "2026-06-10"}]
    p = {"name": "L. Messi", "firstname": "Lionel Andrés", "lastname": "Messi Cuccittini",
         "birth": "1987-06-24"}
    assert pipeline._match_sofifa(p, cands)["overall"] == 86
    assert pipeline._match_sofifa(p | {"name": "X. Other", "firstname": "Xavier",
                                       "lastname": "Other"}, cands) is None
    assert "260037" in pipeline.latest("2026-06-11") and pipeline.latest("2010-01-01") is None
    assert pipeline._iso("1980-5-8") == "1980-05-08" and pipeline._iso("0000-00-00") is None
    assert pipeline.overall("70+2") == 70


def test_players_config_rules():
    fit = {"kind": "fitted", "source": "x"}
    ok = {"name": "elo_poisson", "players": {"source": "sofifa"}, "fit": {},
          "params": {"mu": fit, "slope": fit, "features": fit}}
    run.check({"model": ok})
    with pytest.raises(ValueError, match="together"):
        run.check({"model": ok | {"params": {"mu": fit, "features": fit}}})
    with pytest.raises(ValueError, match="players"):
        run.check({"model": {"name": "elo_poisson", "fit": {}, "params": {"features": fit}}})
