import torch
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Generator


class GraphWrapper:
    def __init__(self, adj: torch.Tensor):
        print(adj)
        self.adj = adj
        edges, self.labels = get_labeled_edges(self.adj)
        self.graph = nx.Graph()
        self.graph.add_edges_from(edges)
        self.pos = nx.spring_layout(self.graph)

    def print(self):
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
                A[i][j + 1::] = torch.randint(6, (1, N - i - 1))
        yield GraphWrapper(A)


def get_labeled_edges(g: torch.Tensor) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], str]]:
    edges: List[Tuple[int, int]] = []
    labels: Dict[Tuple[int, int], str] = {}
    for i in range(g.shape[0]):
        for j in range(i, g.shape[0]):
            if g[i][j] != 0:
                edges.extend([(i + 1, j + 1), (j + 1, i + 1)])
                labels[(i + 1, j + 1)] = str(g[i][j].item())
    return edges, labels


def main():
    for graph in generate_graphs(2, 5):
        graph.print()


if __name__ == "__main__":
    main()
