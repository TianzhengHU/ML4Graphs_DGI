import torch
import torch.nn as nn
import torch.nn.functional as F

# Applies an Diffpool on seq, of shape (batch, nodes, features)
# While taking into account the masking of msk
class Diffpool(nn.Module):
    # Diffpool(node_num, in_features, out_features)
    def __init__(self, node_num, in_features, out_features):
        super(Diffpool, self).__init__()
        self.hidden_1 = 512
        self.hidden_2 = 64
        self.hidden_3 = 1
        # self.hidden_4 = 1
        self.l_1 = nn.Linear(node_num, self.hidden_1)
        self.l_2 = nn.Linear(self.hidden_1, self.hidden_2)
        self.l_3 = nn.Linear(self.hidden_2, self.hidden_3)
        # self.l_4 = nn.Linear(self.hidden_3, self.hidden_4)


    # c = self.read(featuers, adj)
    # it should return a tensor in shape(1,512)
    def forward(self, X, A):
        conv_1 = self.l_1(X.T)
        conv_2 = self.l_2(conv_1)
        conv_3 = self.l_3(conv_2)
        # conv_4 = self.l_4(conv_3)
        # re = torch.sigmoid(conv_3)
        torch.squeeze(conv_3, 1)
        return conv_3.T


# GCN basic operation
class GraphConv(nn.Module):
    # def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False, dropout=0.0, bias=True):
    def __init__(self, input_dim, output_dim):
        super(GraphConv, self).__init__()
        # self.add_self = add_self
        # self.dropout = dropout
        # if dropout > 0.001:
        #     self.dropout_layer = nn.Dropout(p=dropout)
        # self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.normalize_embedding = False
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        # if bias:
        self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        # else:
        #     self.bias = None

    # s = self.diffpool(h_pos, adj)
    def forward(self, x, adj):
        # if self.dropout > 0.001:
        #     x = self.dropout_layer(x)
        x = x.squeeze(dim = 0)
        sparse_x = torch.sparse_coo_tensor(indices=x.nonzero().t(), values=x[x != 0], size=x.size())
        y = torch.matmul(adj, sparse_x)
        # if self.add_self:
        #     y += x
        y = torch.matmul(y, self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
            #print(y[0][0])
        return y

