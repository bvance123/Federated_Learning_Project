"""Evaluation metrics, always computed per hospital.

Owner: Bennett
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def sensitivity(y_true, y_prob, cutoff=0.5):
    """Fraction of truly sick patients the model flags as sick: TP / (TP + FN).

    Returns NaN if there are no sick patients (nothing to catch).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Step 1: turn probabilities into yes/no guesses
    y_guess = (y_prob >= cutoff).astype(int)

    # Step 2: look only at the truly sick patients
    sick = y_true == 1
    n_sick = sick.sum()
    if n_sick == 0:
        return float("nan")

    # Step 3: count the sick patients the guesser missed
    false_negatives = (y_guess[sick] == 0).sum()

    return float((n_sick - false_negatives) / n_sick)


def auroc(y_true, y_prob):
    """Chance a random sick patient gets a higher probability than a random healthy one.

    0.5 = coin flip, 1.0 = perfect. Returns NaN if the sample has only sick or
    only healthy patients (no pairs to compare).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Need at least one sick and one healthy patient to form a pair
    if len(np.unique(y_true)) < 2:
        return float("nan")

    return float(roc_auc_score(y_true, y_prob))
