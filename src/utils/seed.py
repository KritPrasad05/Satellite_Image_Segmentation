import random
import numpy as np
import torch

# SET GLOBAL SEED
def set_seed(seed=42):
    # PYTHON
    random.seed(seed)

    # NUMPY
    np.random.seed(seed)

    # PYTORCH CPU
    torch.manual_seed(seed)

    # PYTORCH CUDA
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"\nSeed set to: {seed}")