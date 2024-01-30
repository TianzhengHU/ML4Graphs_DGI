import torch
import torch.nn as nn

# Applies an average on seq, of shape (batch, nodes, features)
# While taking into account the masking of msk
class Readout(nn.Module):
    def __init__(self):
        super(Readout, self).__init__()

    # c = self.read(h_pos, func, msk=None)
    def forward(self, seq, func):
        if(func=="average"):
            return torch.mean(seq, 1)
        if(func=="set2vec"):
            return
