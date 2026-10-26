
"""Data loading and preprocessing for FLamby Fed-Heart-Disease.

Owner: JJ
"""

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from flamby.datasets.fed_heart_disease import HeartDiseaseRaw


# FLamby center indices: 0 Cleveland, 1 Hungary, 2 Switzerland, 3 Long Beach VA
CENTERS = [0, 1, 2, 3]


def load_center(center, seed):
    """Return (train, val, test) for one hospital.

    Each of train, val and test is a tuple (X, y):
      X: torch.float32 tensor, shape [n, 13], scaled with THIS hospital's train stats only
      y: torch.float32 tensor, shape [n], values 0 or 1

    Center is one of 0, 1, 2, 3 corresponding to Cleveland, Hungary, Switzerland, Long Beach VA.

    Validation is carved from FLamby's training split using 'seed'.
    FLamby's original test split is preserved.
    """
    if center not in CENTERS:
        raise ValueError(f"center must be one of {CENTERS}, got {center}")

    # Keep FLamby's preprocessing but perform our own normalization
    # after carving out validation.
    dataset = HeartDiseaseRaw(normalize=False)

    # Select this center's existing train and test samples.
    train_idx = [
        i for i in range(len(dataset))
        if dataset.centers[i] == center and dataset.sets[i] == "train"
    ]
    test_idx = [
        i for i in range(len(dataset))
        if dataset.centers[i] == center and dataset.sets[i] == "test"
    ]

    if not train_idx or not test_idx:
        raise ValueError(f"No train or test data found for center {center}")

    # Convert the selected samples into tensors.
    X_all = torch.stack(dataset.features)
    y_all = dataset.labels.reshape(-1).to(torch.float32)

    X_trainval = X_all[train_idx]
    y_trainval = y_all[train_idx]
    X_test = X_all[test_idx]
    y_test = y_all[test_idx]

    # Use stratification when every class has enough examples.
    _, class_counts = torch.unique(y_trainval, return_counts=True)
    stratify = (
        y_trainval.numpy()
        if len(class_counts) > 1 and class_counts.min().item() >= 2
        else None
    )

    train_positions, val_positions = train_test_split(
        np.arange(len(train_idx)),
        test_size=0.2,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )

    X_train = X_trainval[train_positions]
    y_train = y_trainval[train_positions]
    X_val = X_trainval[val_positions]
    y_val = y_trainval[val_positions]

    # Fit standardization statistics on the model-training subset only.
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0, unbiased=False)

    # Avoid division by zero for constant features.
    std = torch.where(std > 1e-9, std, torch.ones_like(std))

    # Apply the same training statistics to all three splits.
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    return (
        (X_train.to(torch.float32), y_train.to(torch.float32)),
        (X_val.to(torch.float32), y_val.to(torch.float32)),
        (X_test.to(torch.float32), y_test.to(torch.float32)),
    )