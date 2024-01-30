import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, out_features):
        super(Discriminator,self).__init__()
        # self.score = nn.Bilinear(out_features, out_features, 1)
        # torch.nn.init.xavier_uniform_(self.score.weight.data)
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(out_features, out_features, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    def forward(self, s, h_pos, h_neg):
        s_x = torch.unsqueeze(s, 1)
        s_x = s_x.expand_as(h_pos)

        sc_1 = torch.squeeze(self.f_k(h_pos, s_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_neg, s_x), 2)

        logits = torch.cat((sc_1, sc_2), 1)

        return logits

