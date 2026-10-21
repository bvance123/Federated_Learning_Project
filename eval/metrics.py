"""Evaluation metrics, always computed per hospital.

Owner: Bennett
"""
import csv
import os

import numpy as np
from sklearn.metrics import roc_auc_score

# Shared results CSV format (one row per method, hospital, seed, mu)
COLUMNS = ["method", "center", "seed", "mu", "auroc", "sensitivity", "ece", "brier"]


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


def ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error: how far stated confidence is from what happened.

    Patients are sorted into n_bins equal-width probability bins. In each bin,
    gap = |average probability - fraction really sick|. ECE is the average gap,
    weighted by how many patients are in each bin. 0 = perfectly honest.

    Note: inflated on small test sets (few patients per bin), so compare
    hospitals of different sizes with care. Returns NaN for an empty sample.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    if n == 0:
        return float("nan")

    # Bin index for each patient; probability 1.0 goes in the top bin
    bins = np.minimum((y_prob * n_bins).astype(int), n_bins - 1)

    total = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if not in_bin.any():
            continue  # empty bins don't count
        gap = abs(y_prob[in_bin].mean() - y_true[in_bin].mean())
        total += (in_bin.sum() / n) * gap
    return float(total)


def brier(y_true, y_prob):
    """Average of (probability - truth)^2 over patients. 0 = perfect; no bins needed.

    Squaring stops over- and under-confidence from cancelling and punishes
    confident mistakes hardest. Returns NaN for an empty sample.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean((y_prob - y_true) ** 2))


def evaluate(method, center, seed, y_true, y_prob, mu=None):
    """Score one hospital's exam predictions and return one results row (a dict).

    y_true: real outcomes on this hospital's test set (0/1)
    y_prob: the model's P(disease) for the same patients, same order
    mu:     leash strength, or None for methods without one (left blank in the CSV)
    """
    # Accept torch tensors as well as lists/arrays
    if hasattr(y_true, "detach"):
        y_true = y_true.detach().cpu().numpy()
    if hasattr(y_prob, "detach"):
        y_prob = y_prob.detach().cpu().numpy()
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have one entry per patient")

    return {
        "method": method,
        "center": center,
        "seed": seed,
        "mu": "" if mu is None else mu,
        "auroc": auroc(y_true, y_prob),
        "sensitivity": sensitivity(y_true, y_prob),
        "ece": ece(y_true, y_prob),
        "brier": brier(y_true, y_prob),
    }


def append_rows(rows, path):
    """Add rows to the results CSV, writing the header first if the file is new."""
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
