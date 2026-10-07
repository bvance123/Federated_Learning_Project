"""Shared network used by both the global model and each hospital's personal model.

Default: the FLamby Fed-Heart-Disease baseline, which is logistic regression
(one linear layer, 13 features -> 1 output, sigmoid). Pass hidden > 0 for a
small multi-layer network later on.

Owner: Bennett
"""
import torch
from torch import nn

N_FEATURES = 13  # FLamby Fed-Heart-Disease feature count


class Model(nn.Module):
    def __init__(self, n_features: int = N_FEATURES, hidden: int = 0):
        super().__init__()
        if hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(n_features, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )
        else:
            self.net = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns P(disease) with shape [n]. Train with nn.BCELoss (as FLamby does)."""
        return torch.sigmoid(self.net(x)).squeeze(-1)


def get_weights(model: nn.Module) -> dict:
    """Detached copy of the model's weights (a state_dict)."""
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def set_weights(model: nn.Module, weights: dict) -> None:
    model.load_state_dict(weights)
