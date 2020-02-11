import networkx as nx
import numpy as np
import torch
from G2G.model.graph_wrapper import GraphWrapper
from typing import List, Tuple

from torch.nn import Parameter


def glorot_init(input_dim, output_dim) -> Parameter:
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return Parameter(initial, requires_grad=True)


def shortest_path_length(A: torch.Tensor) -> int:
    return torch.nonzero(A).shape[0]


def reconstructed_matrix_to_shortest_path(a: torch.Tensor, start: int, end: int) -> List[int]:
    assert start > 0 and end > 0
    assert start != end
    start -= 1
    end -= 1

    l = list(a.max(dim=1)[1])
    m: List[int] = [start + 1]

    if end not in l:
        return [0]

    i = l[start]
    c = 0
    while i != end:
        m.append(i.item() + 1)
        i = l[i]
        c += 1
        if c == len(l):
            m.append(0)
            break
    else:
        m.append(end + 1)
    return m


def shortest_as_adj_from_graph_wrapper(g: GraphWrapper, start: int, end: int) -> torch.Tensor:
    A = torch.zeros(g.adj.shape)
    try:
        shortest_path = nx.shortest_path(g.graph, start, end, weight="weight")
        A = shortest_path_to_adj(shortest_path, g.adj.shape[0])
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass
    return A


def adj_to_shortest_path(A: torch.Tensor, start_node: int) -> List[int]:
    nz = {x[0].item() + 1: x[1].item() + 1 for x in torch.nonzero(A)}
    r = [start_node, nz[start_node]] if start_node in nz else [start_node]
    while r[0] != 0 and r[-1] in nz:
        r.append(nz[r[-1]])
    return r


def shortest_path_to_adj(l: List[int], dim: int):
    A = torch.zeros((dim, dim))
    for s, e in zip(l, l[1:]):
        A[s - 1][e - 1] = 1
    return A


def get_combo(dim: int, num: int) -> List[Tuple[int, int]]:
    combo = []
    assert dim > 0 and num > 0
    for _ in range(num):
        start = torch.randint(1, dim + 1, (1, 1)).item()
        end = torch.randint(1, dim + 1, (1, 1)).item()
        while end == start:
            end = torch.randint(1, dim + 1, (1, 1)).item()
        if start > end:
            start, end = end, start
        combo.append((start, end))

    return combo


def prepare_input(start: int, end: int, dim) -> torch.Tensor:
    temp = torch.zeros((dim, dim))
    temp[start - 1] = torch.tensor([1.] * dim)
    temp[:, end - 1] = torch.tensor([1.])
    return temp
