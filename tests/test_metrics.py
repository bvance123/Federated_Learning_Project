import math

import pytest

from eval.metrics import sensitivity

# Small hand-checked example: A, B, C sick; D, E, F healthy
Y_TRUE = [1, 1, 1, 0, 0, 0]
Y_PROB = [0.9, 0.4, 0.7, 0.2, 0.6, 0.1]


def test_sensitivity_hand_example():
    # Catches A and C, misses B
    assert sensitivity(Y_TRUE, Y_PROB) == pytest.approx(2 / 3)


def test_sensitivity_lower_cutoff_catches_more():
    # At cutoff 0.3, B (0.4) is caught too
    assert sensitivity(Y_TRUE, Y_PROB, cutoff=0.3) == 1.0


def test_sensitivity_no_sick_patients_is_nan():
    # Nothing to catch, so the question has no answer
    assert math.isnan(sensitivity([0, 0, 0], [0.2, 0.8, 0.5]))


def test_sensitivity_misses_only_sick_patient():
    # Healthy patients flagged at the cutoff don't count; the one sick patient is missed
    y_true = [0, 0, 0, 0, 1]
    y_prob = [0.3, 0.3, 0.3, 0.3, 0.2]
    assert sensitivity(y_true, y_prob, cutoff=0.3) == 0.0


def test_sensitivity_exactly_at_cutoff_counts_as_caught():
    assert sensitivity([1], [0.5]) == 1.0
