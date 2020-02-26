from random import shuffle
import networkx as nx
import itertools
import torch
from collections import Counter
from G2G.decorators.decorators import logger, Formatter, timer
from G2G.model.graph_wrapper import GraphWrapper
from typing import List, Tuple, Dict
from G2G.model.model import Predictor


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


def get_all_combo(dim: int) -> List[Tuple[int, int]]:
    r: List[Tuple[int, int]] = []
    for combo in itertools.combinations(range(1, dim + 1), r=2):
        r.append(combo)
    shuffle(r)
    return r


def get_combo(max: int, num: int) -> Tuple[int, int]:
    assert max > 0 and num > 0
    for _ in range(num):
        start = torch.randint(1, max + 1, (1, 1)).item()
        end = torch.randint(1, max + 1, (1, 1)).item()
        while end == start:
            end = torch.randint(1, max + 1, (1, 1)).item()
        if start > end:
            start, end = end, start
        yield start, end


def prepare_input(start: int, end: int, dim: int, adj) -> torch.Tensor:
    temp = torch.zeros(*(dim, dim))
    # temp[start - 1, end - 1] += 1
    # temp[start - 1] = torch.tensor([1.] * dim)
    # temp[:, end - 1] = torch.tensor([1.])
    temp[start - 1] = adj[start - 1]
    temp[:, end - 1] = adj[:, end - 1]
    return temp


def is_path_valid(rec, adj):
    for s, e in zip(rec, rec[1:]):
        if adj[s - 1, e - 1] == 0: return False
    return True


@logger(Formatter(lambda x: "Scores:\n" + str([str(k) + ":  " + str(v) + "\n" for k, v in x.items()])))
def get_score(predictor: Predictor, x: List[GraphWrapper], y: Dict[str, Dict[Tuple[int, int], torch.tensor]]) \
        -> Dict[str, float]:
    acc: Dict[str, List[float]] = {'total': [], 'long': [], 'short': [], 'no_path': [], 'invalid_path': []}
    dim = x[0].adj.shape[0]
    tot_len = 0

    for g in x:
        for c1, c2 in itertools.combinations(range(1, dim + 1), r=2):
            tot_len += 1
            label = adj_to_shortest_path(y[str(g)][(c1, c2)], c1)
            rec = reconstructed_matrix_to_shortest_path(predictor(prepare_input(c1, c2, dim, g.adj), g.laplacian), c1,
                                                        c2)
            acc['total'].append(label == rec)
            if len(label) > 2:
                acc['long'].append(label == rec)
            else:
                acc['short'].append(label == rec)
            if 0 in rec:
                acc['no_path'].append(1)
            elif not is_path_valid(rec, g.adj):
                acc['invalid_path'].append(1)

    return {k: sum(v) / tot_len * 100 if k in ("no_path", "invalid_path", "total") else sum(v) / len(v) * 100 for k, v
            in acc.items()}
