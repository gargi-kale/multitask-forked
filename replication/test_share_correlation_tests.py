"""Synthetic checks for share_correlation_tests.py (no model, no data).

    python -m pytest replication/test_share_correlation_tests.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from share_correlation_tests import (bh, boot_p, combine, holm,  # noqa: E402
                                     independent_errors_phi, phi)


def sums(a, b):
    a, b = a.astype(float), b.astype(float)
    return len(a), a.sum(), b.sum(), (a * b).sum()


def correlated_labels(rng, n, rho):
    z = rng.standard_normal((n, 2)) @ np.linalg.cholesky([[1, rho], [rho, 1]]).T
    return z[:, 0] > 0.3, z[:, 1] > -0.2


def test_phi_matches_numpy_corrcoef():
    rng = np.random.default_rng(0)
    a, b = correlated_labels(rng, 5000, 0.5)
    assert np.isclose(phi(*sums(a, b)), np.corrcoef(a, b)[0, 1])


def test_independent_errors_reference_matches_simulation():
    # Flip each head's labels independently at its own rates: the observed phi
    # of the predictions must match the reference, and sit closer to 0 than
    # the label phi (independent mistakes only weaken a correlation).
    rng = np.random.default_rng(1)
    yi, yj = correlated_labels(rng, 200_000, 0.6)
    hi = np.where(yi, rng.random(len(yi)) < 0.8, rng.random(len(yi)) < 0.3)
    hj = np.where(yj, rng.random(len(yj)) < 0.7, rng.random(len(yj)) < 0.2)
    n, si, sj, sy = sums(yi, yj)
    ref = independent_errors_phi(n, si, sj, sy, hi.sum(), hj.sum(),
                                 (hi & yi).sum(), (hj & yj).sum())
    observed = phi(*sums(hi, hj))
    assert abs(observed - ref) < 0.01
    assert 0 < ref < phi(n, si, sj, sy)


def test_shared_cue_pushes_prediction_phi_above_the_reference():
    # Both heads also fire on a shared cue unrelated to either label.
    rng = np.random.default_rng(2)
    yi, yj = correlated_labels(rng, 100_000, 0.2)
    cue = rng.random(len(yi)) < 0.4
    hi, hj = yi | cue, yj | cue
    n, si, sj, sy = sums(yi, yj)
    ref = independent_errors_phi(n, si, sj, sy, hi.sum(), hj.sum(),
                                 (hi & yi).sum(), (hj & yj).sum())
    assert phi(*sums(hi, hj)) > ref + 0.1


def test_perfect_predictions_reproduce_the_label_phi():
    rng = np.random.default_rng(3)
    yi, yj = correlated_labels(rng, 10_000, -0.4)
    n, si, sj, sy = sums(yi, yj)
    ref = independent_errors_phi(n, si, sj, sy, si, sj, si, sj)
    assert np.isclose(ref, phi(n, si, sj, sy))


def test_combine_averages_over_seeds_and_resamples_them():
    per_seed = np.array([[10.0, 1.0, 1.0], [20.0, 3.0, 3.0]])  # (seeds, 1 + B)
    point, draws = combine(per_seed, np.array([[0, 0], [1, 1]]))
    assert point == 15.0 and draws.tolist() == [1.0, 3.0]


def test_boot_p_is_a_two_sided_z_test():
    draws = np.random.default_rng(4).normal(0, 1, 100_000)
    assert np.isclose(boot_p(1.96, draws), 0.05, atol=0.002)


def test_holm_and_bh_known_values():
    ps = [0.01, 0.04, 0.03, 0.20]
    np.testing.assert_allclose(holm(ps), [0.04, 0.09, 0.09, 0.20])
    np.testing.assert_allclose(bh(ps), [0.04, 0.0533333, 0.0533333, 0.20], rtol=1e-5)
