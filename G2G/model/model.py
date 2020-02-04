import torch
import torch.nn.functional as F
from G2G.utils import glorot_init
from torch import nn


class GraphConvolutionLayer(nn.Module):

    def __init__(self, in_features, out_features):
        super(GraphConvolutionLayer, self).__init__()
        self.weight = glorot_init(in_features, out_features)

    def forward(self, x, adj) -> torch.Tensor:
        support = torch.mm(x, self.weight)
        output = torch.mm(adj, support)
        return output  # + self.bias


class Predictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Predictor, self).__init__()
        self.GCN1 = GraphConvolutionLayer(input_dim, output_dim)
        self.GCN2 = GraphConvolutionLayer(output_dim, output_dim)

    def forward(self, x, adj):
        x = F.relu(self.GCN1(x, adj))
        x = self.GCN2(x, adj)
        x = F.softmax(x, dim=0)
        # x = torch.stack([torch.where(a == torch.max(a), torch.max(a), torch.tensor(0.0)) for a in x])
        return x

    @staticmethod
    def loss(A_hat, A):
        return F.binary_cross_entropy(A_hat, A)
