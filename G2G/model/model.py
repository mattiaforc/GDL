import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch import nn


class GraphConvolutionChebychev(nn.Module):

    def __init__(self, F_in, F_out, K, bias):
        super(GraphConvolutionChebychev, self).__init__()
        self.F_in = F_in
        self.F_out = F_out
        self.weight = Parameter(torch.zeros((K, F_in, F_out)), requires_grad=True)
        self.bias = Parameter(torch.zeros(F_out), requires_grad=True) if bias is True else bias
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.shape[1])
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not False: self.bias.data.fill_(0.0)

    def forward(self, x, L):
        Tx_0 = x
        out = torch.mm(Tx_0, self.weight[0])

        if self.weight.size(0) > 1:
            Tx_1 = torch.mm(L, x)
            out = out + torch.matmul(Tx_1, self.weight[1])

        for k in range(2, self.weight.size(0)):
            Tx_2 = 2 * torch.mm(L, Tx_1) - Tx_0
            out = out + torch.matmul(Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not False:
            out = out + self.bias
        return out


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
    def __init__(self, input_dim, output_dim, hidden, k, dropout):
        super(Predictor, self).__init__()
        # self.GCN1 = GraphConvolutionLayer(input_dim, 50)
        # self.GCN2 = GraphConvolutionLayer(50, 50)
        # self.GCN3 = GraphConvolutionLayer(50, 50)
        # self.GCN4 = GraphConvolutionLayer(50, output_dim)
        # self.GCN5 = GraphConvolutionLayer(output_dim, output_dim)
        self.GCN1 = GraphConvolutionChebychev(input_dim, hidden, k, bias=False)
        self.GCN2 = GraphConvolutionChebychev(hidden, hidden, k, bias=False)
        self.GCN3 = GraphConvolutionChebychev(hidden, hidden, k, bias=False)
        self.GCN4 = GraphConvolutionChebychev(hidden, output_dim, k, bias=False)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x, adj):
        x = torch.relu(self.GCN1(x, adj))
        x = self.dropout(torch.relu(self.GCN2(x, adj)))
        x = self.dropout(torch.relu(self.GCN3(x, adj)))
        # x = torch.relu(self.GCN1(x, adj))
        # x = torch.relu(self.GCN2(x, adj))
        # x = torch.relu(self.GCN3(x, adj))
        x = self.GCN4(x, adj)
        x = F.log_softmax(x, dim=1)
        return x

    @staticmethod
    def loss(A_hat, A):
        return F.binary_cross_entropy_with_logits(A_hat, A)


def glorot_init(input_dim, output_dim) -> Parameter:
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return Parameter(initial, requires_grad=True)
