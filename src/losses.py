"""FP32 multilabel ASL; BCE remains in the existing trainer path."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class AsymmetricLoss(nn.Module):
    """Detached asymmetric focusing, clipped negatives, mean over batch/labels.

    Reference: Alibaba-MIIL/ASL AsymmetricLoss (detached focal-weight variant).
    Use log-sigmoid for unbounded stable log probabilities, not log(sigmoid(x)).
    No global grad-mode changes, positive-class weights or target smoothing.
    """
    def __init__(self, gamma_pos=0., gamma_neg=4., clip=.05):
        super().__init__()
        if (not all(math.isfinite(v) for v in (gamma_pos, gamma_neg, clip)) or
                min(gamma_pos, gamma_neg) < 0 or not 0 <= clip < 1):
            raise ValueError("Invalid ASL gamma/clip parameters")
        self.gamma_pos, self.gamma_neg, self.clip = gamma_pos, gamma_neg, clip

    def forward(self, logits, targets):
        with torch.amp.autocast(logits.device.type, enabled=False):
            x, y = logits.float(), targets.float()
            if x.shape != y.shape or x.ndim != 2 or not x.numel():
                raise ValueError("ASL requires nonempty matching (batch, classes) logits/targets")
            if not torch.isfinite(x).all() or not torch.isfinite(y).all() or not ((y == 0) | (y == 1)).all():
                raise ValueError("ASL requires finite logits and binary finite targets")
            positive_log = F.logsigmoid(x)
            negative = (torch.sigmoid(-x) + self.clip).clamp(max=1)
            negative_log = negative.log() if self.clip else F.logsigmoid(-x)
            with torch.no_grad():
                positive_weight = (1 - torch.sigmoid(x)).pow(self.gamma_pos)
                negative_weight = (1 - negative).pow(self.gamma_neg)
            return -(y * positive_weight * positive_log + (1 - y) * negative_weight * negative_log).mean()
