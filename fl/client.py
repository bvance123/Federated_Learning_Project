"""Client side: local training and personal models with mu.

Owner: Nikolay
"""


def local_update(model, global_weights, train_data, mu, epochs):
    """Train one hospital's model starting from the global weights.

    model:          fl.model.Model (a fresh copy; safe to modify)
    global_weights: state_dict of the current global model
    train_data:     (X, y) tensors from load_center: X float [n, 13], y float [n] in {0, 1}
    mu:             drift-penalty strength; loss = BCE + (mu / 2) * ||w - w_global||^2
    epochs:         number of local epochs
    returns:        updated state_dict (use fl.model.get_weights(model))

    Sanity checks:
      mu = 0       -> same as plain local training
      mu very big  -> stays almost identical to global_weights
    """
    # TODO (Nikolay)
    raise NotImplementedError("local_update is not written yet")
