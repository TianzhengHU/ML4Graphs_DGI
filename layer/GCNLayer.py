import torch
import torch.nn as nn
from torch.nn import Parameter

class GCNLayer(nn.Module):
    """
    A single Graph Convolution Layer
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GCNLayer, self).__init__()
        # Define learnable weight matrix
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.act = nn.PReLU()
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, X, A):
        seq_fts = self.fc(X)
        out = torch.unsqueeze(torch.spmm(A, torch.squeeze(seq_fts, 0)), 0)
        out += self.bias

        return self.act(out)