import torch
import torch.nn as nn
import torch.nn.functional as F

# Applies an average on seq, of shape (batch, nodes, features)
# While taking into account the masking of msk
class Readout(nn.Module):
    def __init__(self):
        super(Readout, self).__init__()
        self.sigmoid = nn.Sigmoid()
    # c = self.read(h_pos, func, msk=None)
    def forward(self, h_pos, func):
        if (func == "average"):
            c = torch.mean(h_pos, 1)

        if (func == "sum_norm"):
            sum_1 = torch.sum(h_pos, 1)
            c = F.normalize(sum_1)

        return c

