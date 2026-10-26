"""Training logic for local-only and pooled baselines.

Owner: JJ
"""

import torch

from fl.client import local_update
from fl.data import CENTERS, load_center
from fl.model import Model, get_weights, set_weights


def load_all_centers(seed=0):
    """Load fixed train/validation/test splits for every center."""
    data = {}

    for center in CENTERS:
        train, val, test = load_center(center, seed=seed)
        data[center] = {
            "train": train,
            "val": val,
            "test": test,
        }

    return data


def combine_partitions(data, centers, split):
    """Combine the tensors from the requested centers and split."""
    X = torch.cat([data[c][split][0] for c in centers], dim=0)
    y = torch.cat([data[c][split][1] for c in centers], dim=0)
    return X, y


def run_local_only(data, epochs=50, lr=0.01, batch_size=32):
    """Train one independent model per center."""
    results = {}

    for center in CENTERS:
        torch.manual_seed(0)
        model = Model()
        initial_weights = get_weights(model)

        trained_weights = local_update(
            model=model,
            global_weights=initial_weights,
            train_data=data[center]["train"],
            mu=0.0,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
        )
        set_weights(model, trained_weights)

        results[center] = model

    return results


def run_pooled(data, epochs=50, lr=0.01, batch_size=32):
    """Train one model on all centers' training data."""
    pooled_train = combine_partitions(data, CENTERS, "train")

    torch.manual_seed(0)
    model = Model()
    initial_weights = get_weights(model)

    trained_weights = local_update(
        model=model,
        global_weights=initial_weights,
        train_data=pooled_train,
        mu=0.0,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )
    set_weights(model, trained_weights)

    return model
