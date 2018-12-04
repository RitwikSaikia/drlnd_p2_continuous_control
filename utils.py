import random

import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def to_tensor(x, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x
    return torch.tensor(x, device=device, dtype=dtype)


def set_seed(seed=None):
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
