import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch import nn


class GraphConvolutionLayer(nn.Module):

    def __init__(self, in_features, out_features):
        super(GraphConvolutionLayer, self).__init__()
        self.weight = glorot_init(in_features, out_features)
        self.bias = Parameter(torch.zeros((10, out_features)), requires_grad=True)

    def forward(self, x, adj) -> torch.Tensor:
        support = torch.mm(x, self.weight)
        output = torch.mm(adj, support)
        assert True not in torch.isnan(output)
        return output + self.bias


class Predictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Predictor, self).__init__()
        self.GCN1 = GraphConvolutionLayer(input_dim, 50)
        self.GCN2 = GraphConvolutionLayer(50, 50)
        self.GCN3 = GraphConvolutionLayer(50, 50)
        self.GCN4 = GraphConvolutionLayer(50, 50)
        self.GCN5 = GraphConvolutionLayer(50, 50)
        self.GCN6 = GraphConvolutionLayer(50, output_dim)
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, x, adj):
        x = torch.sigmoid(self.GCN1(x, adj))
        x = torch.sigmoid(self.GCN2(x, adj))
        x = self.dropout(x)
        x = torch.sigmoid(self.GCN3(x, adj))
        x = self.dropout(x)
        # x = torch.sigmoid(self.GCN4(x, adj))
        # x = torch.sigmoid(self.GCN5(x, adj))
        x = self.GCN6(x, adj)
        x = F.log_softmax(x, dim=1)
        return x

    @staticmethod
    def loss(A_hat, A):
        return F.binary_cross_entropy_with_logits(A_hat, A)


def glorot_init(input_dim, output_dim) -> Parameter:
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return Parameter(initial, requires_grad=True)
