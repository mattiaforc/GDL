import torch
import numpy as np
from typing import List
from torch.nn import Parameter
from sklearn.metrics import roc_auc_score, average_precision_score


def glorot_init(input_dim, output_dim) -> Parameter:
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return Parameter(initial, requires_grad=True)


def get_acc(adj_rec, adj_label) -> float:
    labels_all = adj_label.view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy


def get_roc_auc_score(adj_rec, adj_label) -> float:
    print("Rec ", adj_rec[0][0].detach(), "\tOrig ", adj_label[0][0], sep='')
    labels_all = adj_label.view(-1).long()
    preds_all = adj_rec.view(-1).long()
    return roc_auc_score(labels_all, preds_all)


def get_ap_score(adj_rec, adj_label) -> float:
    labels_all = adj_label.view(-1).long()
    preds_all = adj_rec.view(-1).long()
    return average_precision_score(labels_all, preds_all)


def shortest_path_length(A: torch.Tensor) -> int:
    return torch.nonzero(A).shape[0]


def adj_to_shortest_path(A: torch.Tensor, start_node: int) -> List[int]:
    nz = {x[0].item() + 1: x[1].item() + 1 for x in torch.nonzero(A)}
    r = [nz[start_node]] if start_node in nz else [0]
    while r[0] != 0 and r[-1] in nz:
        r.append(nz[r[-1]])
    return r
