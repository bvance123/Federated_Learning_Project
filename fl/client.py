"""Client side: local training and personal models with mu.

Owner: Nikolay
"""

import torch
from torch import nn

from fl.model import get_weights, set_weights


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
    if mu < 0:
      raise ValueError("mu must be non-negative")
    if epochs < 0:
      raise ValueError("epochs must be non-negative")

    X, y = train_data
    parameters = dict(model.named_parameters())
    if not parameters:
      raise ValueError("model must have at least one parameter")
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
      raise ValueError("train_data must contain X [n, features] and y [n]")
    if X.shape[0] == 0:
      raise ValueError("train_data must not be empty")

    first_parameter = next(iter(parameters.values()))
    X = X.to(device=first_parameter.device, dtype=first_parameter.dtype)
    y = y.to(device=first_parameter.device, dtype=first_parameter.dtype)

    set_weights(model, global_weights)
    reference = {
      name: global_weights[name].to(device=parameter.device, dtype=parameter.dtype)
      for name, parameter in parameters.items()
    }

    optimizer = torch.optim.SGD(
      model.parameters(), lr=0.1 / (1.0 + float(mu))
    )
    criterion = nn.BCELoss()
    model.train()

    for _ in range(epochs):
      optimizer.zero_grad(set_to_none=True)
      loss = criterion(model(X), y)
      if mu:
        penalty = sum(
          (parameter - reference[name]).square().sum()
          for name, parameter in parameters.items()
        )
        loss = loss + (float(mu) / 2.0) * penalty
      loss.backward()
      optimizer.step()

    return get_weights(model)
