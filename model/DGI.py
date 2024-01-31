import torch
import torch.nn as nn
from layer import GCNLayer, Readout, Discriminator, Diffpool
class DGI(nn.Module):
    # model = DGI(ft_size, hid_units)
    def __init__(self, node_num, in_features, out_features):
        super(DGI, self).__init__()
        # self.gcn = GCN(n_in, n_h, activation)
        self.gcn = GCNLayer(in_features, out_features)
        self.read = Readout()
        self.diffpool = Diffpool(node_num, in_features, out_features)
        self.sigmoid = nn.Sigmoid()
        self.disc = Discriminator(out_features)


    # logits = model(features, negative_features, adj, sparse = True)
    def forward(self, features, negative_features, adj, func):
        # positive embedding
        h_pos = self.gcn(features, adj)

        # readout function
        if(func == "average"):
            c = self.read(h_pos, func)
            s = self.sigmoid(c)
        if(func == "diffpool"):
            x = h_pos.squeeze(dim = 0)
            c = self.diffpool(x, adj)
            s = self.sigmoid(c)
        if (func == "sum_norm"):
            c = self.read(h_pos, func)
            s = self.sigmoid(c)

        # negative embedding
        h_neg = self.gcn(negative_features, adj)

        # Discriminator
        result = self.disc(s, h_pos, h_neg)

        return result

    def embed(self, seq, adj, func):
        h_1 = self.gcn(seq, adj)

        # readout function
        if (func == "average"):
            c = self.read(h_1, func)
        if (func == "diffpool"):
            x = h_1.squeeze(dim=0)
            c = self.diffpool(x, adj)
        if (func == "sum_norm"):
            c = self.read(h_1, func)

        return h_1.detach(), c.detach()