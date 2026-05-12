import numpy as np
import pytest
from entropypf import (
    get_test_periods,
    get_phases,
    get_entropy,
    get_entropies,
    find_best_period,
)


# ---------------------------------------------------------------------------
# Synthetic light curve (similar to Cincotta et al. 1995, Section 3.1)
# A0=14, A1=-0.5, A2=-0.15, A3=-0.05, T=173.015 days, sigma=0.3
# ---------------------------------------------------------------------------

TRUE_PERIOD = 173.015
RNG = np.random.default_rng(42)


def make_light_curve(n=400, true_period=TRUE_PERIOD, seed=42):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, true_period * 5, n))
    u = (
        14.0
        + (-0.5) * np.sin(2 * np.pi * t / true_period)
        + (-0.15) * np.sin(4 * np.pi * t / true_period)
        + (-0.05) * np.sin(6 * np.pi * t / true_period)
        + rng.uniform(-0.3, 0.3, n)
    )
    return t, u


# ---------------------------------------------------------------------------
# get_test_periods
# ---------------------------------------------------------------------------

class TestGetTestPeriods:
    def test_flat_length(self):
        p = get_test_periods(1, 100, 500, flat=True)
        assert len(p) == 500

    def test_flat_bounds(self):
        p = get_test_periods(1, 100, 500, flat=True)
        assert p[0] == pytest.approx(1.0)
        assert p[-1] == pytest.approx(100.0)

    def test_nonflat_length(self):
        p = get_test_periods(1, 500, 2000)
        assert len(p) == 2000

    def test_nonflat_sorted(self):
        p = get_test_periods(1, 500, 2000)
        assert np.all(np.diff(p) >= 0)

    def test_all_below_threshold_returns_linspace(self):
        p = get_test_periods(1, 20, 100, threshold=30)
        assert len(p) == 100
        np.testing.assert_allclose(p, np.linspace(1, 20, 100))


# ---------------------------------------------------------------------------
# get_phases
# ---------------------------------------------------------------------------

class TestGetPhases:
    def test_phases_in_unit_interval(self):
        t, u = make_light_curve()
        phi = get_phases(t, u, TRUE_PERIOD)
        assert np.all(phi >= 0) and np.all(phi < 1)

    def test_phases_length_matches_input(self):
        t, u = make_light_curve(n=200)
        phi = get_phases(t, u, TRUE_PERIOD)
        assert len(phi) == 200


# ---------------------------------------------------------------------------
# get_entropy
# ---------------------------------------------------------------------------

class TestGetEntropy:
    def test_uniform_distribution_is_max(self):
        # Uniform Mu → entropy ≈ log(L*K)
        L, K = 8, 8
        Mu = np.full((L, K), 1.0 / (L * K))
        S = get_entropy(Mu)
        assert S == pytest.approx(np.log(L * K), rel=1e-6)

    def test_single_cell_is_zero(self):
        Mu = np.zeros((4, 4))
        Mu[0, 0] = 1.0
        assert get_entropy(Mu) == pytest.approx(0.0)

    def test_empty_matrix_is_zero(self):
        assert get_entropy(np.zeros((4, 4))) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# get_entropies
# ---------------------------------------------------------------------------

class TestGetEntropies:
    def test_output_length(self):
        t, u = make_light_curve()
        periods = np.linspace(50, 400, 300)
        S = get_entropies(t, u, periods)
        assert len(S) == 300

    def test_values_in_zero_one(self):
        t, u = make_light_curve()
        periods = np.linspace(50, 400, 300)
        S = get_entropies(t, u, periods)
        assert np.all(S >= 0) and np.all(S <= 1)

    def test_minimum_near_true_period(self):
        """Entropy should be lower at the true period than at a random one."""
        t, u = make_light_curve()
        periods = np.array([TRUE_PERIOD, TRUE_PERIOD * 1.37])
        S = get_entropies(t, u, periods)
        assert S[0] < S[1]


# ---------------------------------------------------------------------------
# find_best_period
# ---------------------------------------------------------------------------

class TestFindBestPeriod:
    def test_recovers_true_period(self):
        """Top candidate should be within 1% of the true period."""
        t, u = make_light_curve(n=400)
        periods, entropies = find_best_period(t, u, p0=100, p1=300, p_num=2000)
        assert abs(periods[0] - TRUE_PERIOD) / TRUE_PERIOD < 0.01

    def test_returns_arrays(self):
        t, u = make_light_curve()
        periods, entropies = find_best_period(t, u, p0=100, p1=300, p_num=500)
        assert isinstance(periods, np.ndarray)
        assert isinstance(entropies, np.ndarray)

    def test_entropies_sorted_ascending(self):
        t, u = make_light_curve()
        _, entropies = find_best_period(t, u, p0=100, p1=300, p_num=500,
                                        n_candidates=3)
        assert np.all(np.diff(entropies) >= 0)

    def test_n_candidates_respected(self):
        t, u = make_light_curve()
        periods, entropies = find_best_period(t, u, p0=100, p1=300, p_num=500,
                                              n_candidates=2)
        assert len(periods) <= 2
        assert len(entropies) <= 2
