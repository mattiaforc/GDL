import torch
import networkx as nx
import matplotlib.pyplot as plt
import itertools
from typing import List, Tuple, Dict, Generator
from tqdm import tqdm


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


def generate_graphs(iterations: int, N: int) -> Generator[GraphWrapper, None, None]:
    for _ in range(iterations):
        A = torch.zeros((N, N))
        for i in range(N):
            for j in range(i + 1):
                A[i][j] = A[j][i]
            else:
                A[i][j + 1::] = torch.nn.functional.relu(torch.randn((1, N - i - 1)))
        yield GraphWrapper(A)


def get_labeled_edges(g: torch.Tensor) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], str]]:
    edges: List[Tuple[int, int]] = []
    labels: Dict[Tuple[int, int], str] = {}
    for i in range(g.shape[0]):
        for j in range(g.shape[0]):
            if g[i][j] != 0:
                edges.extend([(i + 1, j + 1), (j + 1, i + 1)])
                labels[(i + 1, j + 1)] = str(g[i][j].item())[0:4]
    return edges, labels


def shortest_path_as_adjacency_matrix(g: GraphWrapper, start: int, end: int) -> torch.Tensor:
    A = torch.zeros(g.adj.shape)
    try:
        shortest_path = nx.shortest_path(g.graph, start, end, weight="weight")
        for s, e in zip(shortest_path, shortest_path[1:]):
            A[s - 1][e - 1] = g.adj[s - 1][e - 1]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass
    return A


def generate_dataset(iterations: int, N: int) \
        -> Tuple[List[GraphWrapper], Dict[GraphWrapper, Dict[Tuple[int, int], torch.Tensor]]]:
    y = {}
    x = []
    for graph in tqdm(generate_graphs(iterations, N), total=iterations):
        x.append(graph)
        y[graph] = {}
        for combo in itertools.combinations(range(N), r=2):
            y[graph][combo] = shortest_path_as_adjacency_matrix(graph, *combo)
    return x, y
