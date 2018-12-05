from torch import Tensor
import numpy as np
import torch
from finegrain import refine

class AverageMeter:
    """
    Class to be an average meter for any average metric like loss, accuracy, etc..
    """

    def __init__(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def val(self):
        return self.avg


def apk(actual, predicted, k=3):
    """
    Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk3(pred:Tensor, targ:Tensor, mode="", idx=0):
    pred = pred.detach()
    order = np.argsort(pred.cpu().numpy(), 1)[:, -3:]
#    if mode != "":
#        order = refine(order, mode, idx)

    order = [[z.item(),y.item(),x.item()] for x,y, z in order]

    actual = targ.cpu().numpy()
    predicted = order

    r = np.mean([apk([a], p, 3) for a, p in zip(actual, predicted)])
    return torch.tensor(r)
