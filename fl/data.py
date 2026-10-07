"""Data loading and preprocessing for FLamby Fed-Heart-Disease.

Owner: JJ
"""

# FLamby center indices: 0 Cleveland, 1 Hungary, 2 Switzerland, 3 Long Beach VA
CENTERS = [0, 1, 2, 3]


def load_center(center, seed):
    """Return (train, val, test) for one hospital.

    Each of train, val and test is a tuple (X, y):
      X: torch.float32 tensor, shape [n, 13], scaled with THIS hospital's train stats only
      y: torch.float32 tensor, shape [n], values 0 or 1

    val is carved from FLamby's train split using `seed`; test is FLamby's test split.
    """
    # TODO (JJ)
    raise NotImplementedError("load_center is not written yet")
