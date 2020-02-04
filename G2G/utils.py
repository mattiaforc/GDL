import torch
import numpy as np
from typing import List
from torch.nn import Parameter
from sklearn.metrics import roc_auc_score, average_precision_score


def glorot_init(input_dim, output_dim) -> Parameter:
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return Parameter(initial, requires_grad=True)


def shortest_path_length(A: torch.Tensor) -> int:
    return torch.nonzero(A).shape[0]


def adj_to_shortest_path(A: torch.Tensor, start_node: int) -> List[int]:
    nz = {x[0].item() + 1: x[1].item() + 1 for x in torch.nonzero(A)}
    r = [nz[start_node]] if start_node in nz else [0]
    while r[0] != 0 and r[-1] in nz:
        r.append(nz[r[-1]])
    return r
