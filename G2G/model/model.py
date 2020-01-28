import torch
import math
import torch.nn.functional as F
from G2G.utils import glorot_init, get_acc, get_ap_score, get_roc_auc_score
from torch import nn
from torch.nn.parameter import Parameter


class GAE(nn.Module):
    def __init__(self, nfeat, nhid, dimz):
        super(GAE, self).__init__()
        self.nhid = nhid
        self.w = Parameter(torch.FloatTensor(self.nhid, self.nhid))

    def encode(self, adj) -> torch.Tensor:
        return torch.mm(adj, self.w)

    def decode(self, z) -> torch.Tensor:
        return torch.mm(z, z.t())

    def forward(self, adj) -> torch.Tensor:
        z = self.encode(adj)
        A_recon = self.decode(z)
        return A_recon

    @staticmethod
    def loss(A_hat, A, norm, weights) -> torch.Tensor:
        cross_entropy_loss = F.cross_entropy(A_hat, A, weight=weights)
        # bce_loss = norm * F.binary_cross_entropy_with_logits(A_hat.view(-1), A.view(-1))
        # kl_loss = 0.5 / A_hat.size(0) * (1 + 2 * self.log_std - self.mean.pow(2) - torch.exp(self.log_std).pow(2)).sum(1).mean()
        # bce_loss = norm * F.mse_loss(A_hat.view(-1), A.view(-1), reduction="mean")
        return cross_entropy_loss  # - kl_loss


class GraphConvolutionLayer(nn.Module):

    def __init__(self, in_features, out_features):
        super(GraphConvolutionLayer, self).__init__()
        self.weight = glorot_init(in_features, out_features)

    def forward(self, x, adj) -> torch.Tensor:
        support = torch.mm(x, self.weight)
        output = torch.mm(adj, support)
        return output  # + self.bias


class GraphConvolutionChebychev(nn.Module):

    def __init__(self, F_in, F_out, K):
        super(GraphConvolutionChebychev, self).__init__()
        self.F_in = F_in
        self.F_out = F_out
        self.weight = Parameter(torch.FloatTensor(K, self.F_in, self.F_out))
        self.bias = Parameter(torch.FloatTensor(self.F_out))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.fill_(0.0)

    def forward(self, x, L) -> torch.Tensor:
        Tx_0 = x
        out = torch.mm(Tx_0, self.weight[0])
        if self.weight.size(0) > 1:
            Tx_1 = torch.mm(L, x)
            out = out + torch.matmul(Tx_1, self.weight[1])
        for k in range(2, self.weight.size(0)):
            Tx_2 = 2 * torch.mm(L, Tx_1) - Tx_0
            out = out + torch.matmul(Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2
        return out + self.bias if self.bias is not None else out


class ChebychevConvolutionalNetwork(nn.Module):

    def __init__(self, net_parameters):
        super(ChebychevConvolutionalNetwork, self).__init__()
        self.F_in, self.nclasses, self.K, self.dropout, self.hidden1 = net_parameters
        self.GCN1 = GraphConvolutionChebychev(self.F_in, self.hidden1, self.K)
        self.GCN2 = GraphConvolutionChebychev(self.hidden1, self.nclasses, self.K)

    def forward(self, x, L) -> torch.Tensor:
        x = self.GCN1(x, L)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.GCN2(x, L)
        return F.log_softmax(x, dim=1)

    def loss(self, y, y_target, l2_regularization) -> torch.Tensor:
        loss = nn.CrossEntropyLoss()(y, y_target)
        l2_loss = 0.0
        for param in self.parameters():
            data = param * param
            l2_loss += data.sum()
        loss += 0.5 * l2_regularization * l2_loss
        return loss

    """
    def test(self):
        self.eval()
        _, pred = self(data.x, M_test).max(dim=1)
        correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
        acc = correct / data.test_mask.sum().item()
        print('Accuracy: {:.4f}'.format(acc))
    """
