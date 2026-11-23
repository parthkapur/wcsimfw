import torch

from wcsimfw import match, run, squad


def test_windows_end_before_the_next_roster_and_the_origin():
    w = squad.windows("2026-06-11")
    assert w[0] == ("180076", "2018-06-11", "2019-06-11")
    assert w[1] == ("230006", "2022-11-16", "2023-11-16")
    assert w[2] == ("240043", "2024-06-12", "2025-06-12")
    assert w[3] == ("260037", "2026-06-10", "2026-06-11")  # one day: no tournament match in it
    assert all(start < end <= "2026-06-11" for _, start, end in w)
    assert squad.windows("2023-01-01")[1][2] == "2023-01-01"


def test_latest_is_dated_before():
    assert squad.latest("2026-06-11") == "260037"
    assert squad.latest("2026-06-10") == "240043"


def test_overall_with_live_boost():
    assert squad._overall("70+2") == 70 and squad._overall("81") == 81


def test_fit_recovers_squad_weight():
    g = torch.Generator().manual_seed(1)
    n = 30_000
    gap = torch.randn(n, generator=g, dtype=torch.float64) * 300
    dq = 0.02 * gap + torch.randn(n, generator=g, dtype=torch.float64) * 4  # related, not equal
    home = torch.zeros(n, dtype=torch.float64)
    e = 0.4 * gap / 400 + 0.04 * dq
    lam = torch.tensor(1.1).log()
    fit = match.fit_elo_poisson(gap, home, torch.poisson((lam + e).exp(), g),
                                torch.poisson((lam - e).exp(), g), squad_gap=dq)
    for k, true in (("slope", 0.4), ("squad", 0.04), ("squad_weight", 40.0)):
        value, se = fit[k]
        assert abs(value - true) < 4 * se, k


def test_squad_option_needs_its_weight():
    import pytest
    base = {"name": "elo_poisson", "params": {"mu": {"value": 1.3, "kind": "fixed",
                                                     "source": "x"}}}
    with pytest.raises(ValueError, match="go together"):
        run.check({"model": base | {"squad": "sofifa"}})
