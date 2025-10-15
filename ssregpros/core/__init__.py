import numpy as np
import random
import torch


def set_deterministic_seed(seed: int | None):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
