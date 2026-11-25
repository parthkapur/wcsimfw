import pytest
import torch

from wcsimfw import elo, match, tournament


def game(gh, ga, competition="FIFA World Cup", neutral=True):
    return {"home": "A", "away": "B", "goals": (gh, ga), "competition": competition,
            "neutral": neutral}


def test_update_rules():
    r = {}
    assert elo.update(r, game(1, 0)) == 0 and r == {"A": 1530, "B": 1470}  # K 60 x 1/2
    r = {}
    elo.update(r, game(3, 0, "Friendly"))
    assert r["A"] == pytest.approx(1500 + 20 * 14 / 8 * 0.5)
    r = {}
    elo.update(r, game(0, 0, neutral=False))  # home side expected to win: a draw costs it
    assert r["A"] < 1500 < r["B"] and r["A"] + r["B"] == pytest.approx(3000)


def test_k_factor():
    assert [elo.k_factor(c) for c in ("FIFA World Cup", "Copa América",
                                      "UEFA Euro qualification", "UEFA Nations League",
                                      "Friendly", "Kirin Cup")] == [60, 50, 40, 40, 20, 30]


def test_tournament_matches_count_extra_time():
    t = tournament.Tournament.load()
    rows = elo.tournament_matches(t.played("final"))
    assert len(rows) == 103 and rows == sorted(rows, key=lambda m: m["date"])
    assert len(elo.tournament_matches(t.played("md1"))) == 24


def test_fit_recovers_known_rates():
    g = torch.Generator().manual_seed(0)
    gap = torch.randn(20_000, generator=g, dtype=torch.float64) * 300
    home = (torch.rand(20_000, generator=g) < 0.5).double()
    lh = torch.log(torch.tensor(1.2)) + 0.7 * gap / 400 + 0.25 * home
    la = torch.log(torch.tensor(1.2)) - 0.7 * gap / 400
    fit = match.fit_elo_poisson(gap, home, torch.poisson(lh.exp(), g), torch.poisson(la.exp(), g))
    for k, true in (("mu", 1.2), ("slope", 0.7), ("home", 0.25)):
        value, se = fit[k]
        assert abs(value - true) < 4 * se and se < 0.05, k
