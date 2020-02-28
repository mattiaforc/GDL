import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch import nn


class GraphConvolutionLayer(nn.Module):

    def __init__(self, in_features, out_features):
        super(GraphConvolutionLayer, self).__init__()
        self.weight = glorot_init(in_features, out_features)
        self.bias = Parameter(torch.zeros((in_features, out_features)), requires_grad=True)

    def forward(self, x, adj) -> torch.Tensor:
        support = torch.mm(x, self.weight)
        output = torch.mm(adj, support)
        assert True not in torch.isnan(output)
        return output + self.bias


class Predictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Predictor, self).__init__()
        self.GCN1 = GraphConvolutionLayer(input_dim, output_dim)
        self.GCN2 = GraphConvolutionLayer(output_dim, output_dim)
        self.GCN3 = GraphConvolutionLayer(output_dim, output_dim)
        self.GCN4 = GraphConvolutionLayer(output_dim, output_dim)
        self.GCN5 = GraphConvolutionLayer(output_dim, output_dim)

    def forward(self, x, adj):
        x = torch.sigmoid(self.GCN1(x, adj))
        # x = self.GCN1(x, adj)
        x = torch.sigmoid(self.GCN2(x, adj))
        x = torch.sigmoid(self.GCN3(x, adj))
        x = torch.sigmoid(self.GCN4(x, adj))
        x = self.GCN5(x, adj)
        x = F.softmax(x, dim=1)
        return x

    @staticmethod
    def loss(A_hat, A):
        return F.binary_cross_entropy(A_hat, A)


def glorot_init(input_dim, output_dim) -> Parameter:
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return Parameter(initial, requires_grad=True)
