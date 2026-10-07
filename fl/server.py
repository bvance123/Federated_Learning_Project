"""Server side: FedAvg aggregation and the round loop.

Owner: Bennett
"""
import copy

import torch

from fl.model import get_weights, set_weights


def aggregate(weights_list: list, sample_counts: list) -> dict:
    """FedAvg: average of client weights, weighted by each client's sample count.

    weights_list:  list of state_dicts (one per hospital), all with the same keys
    sample_counts: list of ints, number of training samples per hospital
    returns:       new global state_dict
    """
    if len(weights_list) != len(sample_counts):
        raise ValueError("weights_list and sample_counts must be the same length")
    total = float(sum(sample_counts))
    if total <= 0:
        raise ValueError("sample_counts must sum to a positive number")

    new_weights = {}
    for key in weights_list[0]:
        new_weights[key] = sum(
            w[key].float() * (n / total) for w, n in zip(weights_list, sample_counts)
        )
    return new_weights


def drift(weights: dict, global_weights: dict) -> float:
    """L2 distance ||w - w_global|| across all parameters."""
    sq = sum(((weights[k].float() - global_weights[k].float()) ** 2).sum() for k in weights)
    return float(torch.sqrt(sq))


def run_rounds(global_model, clients, local_update, n_rounds, mu, epochs, verbose=True):
    """Run FedAvg for n_rounds.

    global_model: fl.model.Model instance (updated in place)
    clients:      list of dicts: {"name": str, "train": (X, y)}
    local_update: function(model, global_weights, train_data, mu, epochs) -> updated weights
    returns:      history, a list with one dict per round:
                  {"round": r, "drift": {client_name: float}, "mean_drift": float}
    """
    history = []
    for r in range(n_rounds):
        global_weights = get_weights(global_model)
        updates, counts, drifts = [], [], {}

        for c in clients:
            local_model = copy.deepcopy(global_model)  # each client starts from a fresh copy
            new_w = local_update(local_model, global_weights, c["train"], mu, epochs)
            updates.append(new_w)
            counts.append(len(c["train"][1]))
            drifts[c["name"]] = drift(new_w, global_weights)

        set_weights(global_model, aggregate(updates, counts))
        mean_drift = sum(drifts.values()) / len(drifts)
        history.append({"round": r, "drift": drifts, "mean_drift": mean_drift})
        if verbose:
            print(f"round {r:3d}  mean drift {mean_drift:.4f}")
    return history


if __name__ == "__main__":
    # End-to-end FedAvg on the real data. Works once data.py and client.py are filled in.
    from fl.client import local_update
    from fl.data import CENTERS, load_center
    from fl.model import Model

    torch.manual_seed(0)
    clients = []
    for center in CENTERS:
        train, val, test = load_center(center, seed=0)
        clients.append({"name": f"center_{center}", "train": train})

    model = Model()
    run_rounds(model, clients, local_update, n_rounds=20, mu=0.0, epochs=1)
