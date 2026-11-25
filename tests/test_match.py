import pytest
import torch

from wcsimfw import match

T = torch.tensor


def test_matrix_shape_and_mass():
    m = match.poisson(T([0.5, 1.3, 3.0], dtype=torch.float64), T(1.3, dtype=torch.float64))
    assert m.shape == (3, match.GOALS, match.GOALS)
    assert torch.allclose(m.sum((-2, -1)), torch.ones(3, dtype=torch.float64))
    assert (m >= 0).all()


def test_mean_goals_equal_rate():
    m = match.poisson(T(1.7, dtype=torch.float64), T(0.4, dtype=torch.float64))
    g = torch.arange(match.GOALS, dtype=torch.float64)
    assert (m.sum(-1) @ g).item() == pytest.approx(1.7, abs=1e-5)  # tail above 10 counted as 10
    assert (m.sum(-2) @ g).item() == pytest.approx(0.4, abs=1e-5)


def test_zero_rate_means_no_goals():
    m = match.poisson(T(0.0, dtype=torch.float64), T(1.0, dtype=torch.float64))
    assert not m.isnan().any() and m[0].sum() == pytest.approx(1.0)


def test_equal_elo_is_even_and_rate_is_mu():
    rh, ra = match.elo_rates(T(1800.0), T(1800.0))
    assert rh.item() == pytest.approx(match.MU) and ra.item() == pytest.approx(match.MU)
    p = match.outcome(match.elo_poisson(T(1800.0), T(1800.0)))
    assert p[0].item() == pytest.approx(p[2].item())


def test_swap_reverses_outcome_and_gap_helps():
    elo = T([2100.0, 1900.0, 1700.0], dtype=torch.float64)
    p = match.outcome(match.elo_poisson(elo, T(1700.0, dtype=torch.float64)))
    q = match.outcome(match.elo_poisson(T(1700.0, dtype=torch.float64), elo))
    assert torch.allclose(p, q.flip(-1))
    assert torch.allclose(p.sum(-1), torch.ones(3, dtype=torch.float64))
    assert p[0, 0] > p[1, 0] > p[2, 0]


def test_advance():
    def tie(a, b):
        rh, ra = match.elo_rates(T(a, dtype=torch.float64), T(b, dtype=torch.float64))
        x = match.EXTRA_TIME
        return match.advance(match.poisson(rh, ra), match.poisson(rh * x, ra * x))

    assert tie(1800.0, 1800.0).item() == pytest.approx(0.5)
    assert (tie(2000.0, 1800.0) + tie(1800.0, 2000.0)).item() == pytest.approx(1.0)
    p = match.outcome(match.elo_poisson(T(2000.0, dtype=torch.float64),
                                        T(1800.0, dtype=torch.float64)))
    assert p[0] < tie(2000.0, 1800.0) < p[0] + p[1]


def test_host_bonus_only_in_its_place():
    elo = T([1800.0, 1800.0], dtype=torch.float64)
    host = T([[0.0, 0.0], [0.5, 0.0]], dtype=torch.float64)  # team 0 hosts place 1
    full, adv = match.elo_pairs(elo, host=host)
    neutral, tie = match.elo_pairs(elo)
    assert torch.allclose(full[0], neutral) and torch.allclose(adv[0], tie)
    g = torch.arange(match.GOALS, dtype=torch.float64)
    assert (full[1, 0, 1].sum(-1) @ g).item() == pytest.approx(match.MU * 1.64872, rel=1e-4)
    assert (full[1, 1, 0].sum(-2) @ g).item() == pytest.approx(match.MU * 1.64872, rel=1e-4)
    assert adv[1, 0, 1] > 0.5 and (adv[1, 0, 1] + adv[1, 1, 0]).item() == pytest.approx(1.0)


def test_level_learns_and_fit_level_recovers_the_prior():
    assert match.level(0, 0, 100) == 1 and match.level(150, 100, 100) == pytest.approx(1.25)
    g = torch.Generator().manual_seed(4)
    torch.manual_seed(4)
    f = torch.distributions.Gamma(100.0, 100.0).sample((4000,)).double()
    e = torch.full((4000,), 120.0, dtype=torch.float64)
    k = match.fit_level(torch.poisson(f * e, g), e)
    assert 85 < k < 118
    assert match.fit_level(torch.poisson(e[:200], g), e[:200]) > 1000  # no spread: history only


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_matches_cpu():
    elo = T([2000.0, 1800.0])
    gpu = match.elo_poisson(elo.cuda(), elo.flip(0).cuda())
    assert torch.allclose(gpu.cpu(), match.elo_poisson(elo, elo.flip(0)), atol=1e-6)
