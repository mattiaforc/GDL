from typing import Tuple, List, Dict
import networkx as nx
import torch
from matplotlib import pyplot as plt


class GraphWrapper:
    def __init__(self, adj: torch.Tensor, pos=None):
        self.adj = adj
        edges, self.labels = get_labeled_edges(self.adj)
        self.graph = nx.Graph()
        self.graph.add_edges_from(edges)
        nx.set_edge_attributes(self.graph, {k: float(v) for k, v in self.labels.items()}, name='weight')
        self.pos = pos or nx.spring_layout(self.graph)

    def print(self):
        print(self.adj)
        nx.draw_networkx_edge_labels(self.graph, self.pos, self.labels)
        nx.draw_networkx_nodes(self.graph, self.pos)
        nx.draw_networkx_edges(self.graph, self.pos)
        nx.draw_networkx_labels(self.graph, self.pos)
        plt.show()


def get_labeled_edges(g: torch.Tensor) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], str]]:
    edges: List[Tuple[int, int]] = []
    labels: Dict[Tuple[int, int], str] = {}
    for i in range(g.shape[0]):
        for j in range(g.shape[0]):
            if g[i][j] != 0:
                edges.extend([(i + 1, j + 1), (j + 1, i + 1)])
                labels[(i + 1, j + 1)] = str(g[i][j].item())[0:4]
    return edges, labels
