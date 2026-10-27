import csv
import math

import pytest
import torch

from eval.metrics import COLUMNS, append_rows, auroc, brier, ece, evaluate, sensitivity

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


def test_auroc_hand_example():
    # 9 sick-healthy pairs; only B (0.4) vs E (0.6) is ranked wrong
    assert auroc(Y_TRUE, Y_PROB) == pytest.approx(8 / 9)


def test_auroc_perfect_ranking():
    assert auroc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0


def test_auroc_backwards_ranking():
    # Every healthy patient ranked above every sick one
    assert auroc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0


def test_auroc_one_class_is_nan():
    assert math.isnan(auroc([0, 0, 0], [0.2, 0.8, 0.5]))
    assert math.isnan(auroc([1, 1, 1], [0.2, 0.8, 0.5]))


def test_ece_hand_example():
    # Low bin: 4 patients told 0.2, 1 sick (gap 0.05)
    # High bin: 6 patients told 0.8, 3 sick (gap 0.3)
    # ECE = 0.4 * 0.05 + 0.6 * 0.3 = 0.20
    y_true = [1, 0, 0, 0] + [1, 1, 1, 0, 0, 0]
    y_prob = [0.2] * 4 + [0.8] * 6
    assert ece(y_true, y_prob, n_bins=2) == pytest.approx(0.20)


def test_ece_honest_guesser_is_zero():
    # Told 0.5, and exactly half were sick
    assert ece([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.0)


def test_ece_single_patient_is_inflated():
    # Even an honest "0.7" scores a big gap with one patient: the small-sample problem
    assert ece([1], [0.7]) == pytest.approx(0.3)
    assert ece([0], [0.7]) == pytest.approx(0.7)


def test_ece_probability_one_lands_in_top_bin():
    assert ece([1], [1.0]) == pytest.approx(0.0)


def test_ece_empty_is_nan():
    assert math.isnan(ece([], []))


def test_brier_hand_example():
    # ((0.8 - 1)^2 + (0.3 - 0)^2) / 2 = (0.04 + 0.09) / 2
    assert brier([1, 0], [0.8, 0.3]) == pytest.approx(0.065)


def test_brier_perfect_confident_guesser_is_zero():
    assert brier([1, 0, 1], [1.0, 0.0, 1.0]) == 0.0


def test_brier_empty_is_nan():
    assert math.isnan(brier([], []))


def test_evaluate_builds_row_from_hand_example():
    row = evaluate("local_only", 2, 0, Y_TRUE, Y_PROB)
    assert list(row) == COLUMNS
    assert row["method"] == "local_only" and row["center"] == 2 and row["seed"] == 0
    assert row["n_test"] == 6  # six patients in the hand example
    assert row["mu"] == ""  # no leash for local-only
    assert row["auroc"] == pytest.approx(8 / 9)
    assert row["sensitivity"] == pytest.approx(2 / 3)


def test_evaluate_accepts_torch_tensors():
    row = evaluate("ditto", 0, 1, torch.tensor(Y_TRUE).float(), torch.tensor(Y_PROB), mu=0.1)
    assert row["mu"] == 0.1
    assert row["auroc"] == pytest.approx(8 / 9)


def test_evaluate_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        evaluate("local_only", 0, 0, [1, 0], [0.5])


def test_append_rows_writes_header_once(tmp_path):
    path = tmp_path / "results.csv"
    append_rows([evaluate("local_only", 0, 0, Y_TRUE, Y_PROB)], path)
    append_rows([evaluate("local_only", 1, 0, Y_TRUE, Y_PROB)], path)
    with open(path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2  # header written only once
    assert [r["center"] for r in rows] == ["0", "1"]
