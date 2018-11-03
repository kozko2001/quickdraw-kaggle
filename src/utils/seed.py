import random
import numpy as np
import torch

def use_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("seed ", seed)


if __name__ == "__main__":
    use_seed(3333)
