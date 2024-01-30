import torch
import torch.nn as nn
from layer import GCNLayer, Readout, Discriminator
class DGI(nn.Module):
    def __init__(self, in_features, out_features):
        super(DGI, self).__init__()
        # self.gcn = GCN(n_in, n_h, activation)
        self.gcn = GCNLayer(in_features, out_features)
        self.read = Readout()
        self.sigmoid = nn.Sigmoid()
        self.disc = Discriminator(out_features)


    # logits = model(features, negative_features, adj, sparse = True)
    def forward(self, features, negative_features, adj, func):
        h_pos = self.gcn(features, adj)
        c = self.read(h_pos, func)
        s = self.sigmoid(c)

        h_neg = self.gcn(negative_features, adj)

        result = self.disc(s, h_pos, h_neg)

        return result

        # Detach the return variables

    def embed(self, seq, adj, func):
        h_1 = self.gcn(seq, adj)
        c = self.read(h_1, func)

        return h_1.detach(), c.detach()