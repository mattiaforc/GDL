from collections import Counter
import networkx as nx
import torch
import uuid
from typing import Tuple, List, Dict
from matplotlib import pyplot as plt


def degree_matrix(A: torch.Tensor, device: torch.device) -> torch.Tensor:
    degrees = list(Counter([v.item() for v in torch.nonzero(A, as_tuple=True)[0]]).values())
    D = torch.zeros_like(A, device=device)
    for i in range(A.shape[0]):
        D[i, i] = degrees[i]
    return D


def laplacian(A: torch.Tensor, device: torch.device) -> torch.Tensor:
    # I + D **-1/2 A D ** -1/2 -> D (of this A) ** -1/2 (A + I) D (of this A) ** -1/2
    A = A + torch.eye(A.shape[0], device=device)
    D = degree_matrix(A, device=device)
    D = torch.where(D != 0, D.pow(-1 / 2), torch.tensor(0., device=device))
    L = torch.mm(D, torch.mm(A, D))
    assert True not in torch.isnan(L)
    return L


class GraphWrapper:
    def __init__(self, adj: torch.Tensor, device: torch.device, pos=None):
        self.adj = adj
        self.laplacian = laplacian(adj, device=device)
        edges, self.labels = get_labeled_edges(self.adj)
        self.graph = nx.Graph()
        self.graph.add_edges_from(edges)
        nx.set_edge_attributes(self.graph, {k: float(v) for k, v in self.labels.items()}, name='weight')
        self.pos = pos or nx.spring_layout(self.graph)
        self.__uuid = uuid.uuid4()

    def print(self, print_adj: bool = False):
        if print_adj: print(self.adj)
        nx.draw_networkx_edge_labels(self.graph, self.pos, self.labels)
        nx.draw_networkx_nodes(self.graph, self.pos)
        nx.draw_networkx_edges(self.graph, self.pos)
        nx.draw_networkx_labels(self.graph, self.pos)
        plt.show()

    def __hash__(self):
        return self.__uuid.__hash__()

    def __eq__(self, other):
        return isinstance(other, GraphWrapper) and self.__uuid == other.__uuid

    def __str__(self):
        return str(self.__uuid)


def get_labeled_edges(g: torch.Tensor) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], str]]:
    edges: List[Tuple[int, int]] = []
    labels: Dict[Tuple[int, int], str] = {}
    for i in range(g.shape[0]):
        for j in range(g.shape[0]):
            if g[i][j] != 0:
                edges.extend([(i + 1, j + 1), (j + 1, i + 1)])
                labels[(i + 1, j + 1)] = str(g[i][j].item())[0:4]
    return edges, labels
