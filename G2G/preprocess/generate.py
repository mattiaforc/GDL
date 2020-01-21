import torch
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict


def generate_graphs(iterations: int, N: int) -> torch.Tensor:
    for _ in range(iterations):
        A = torch.zeros((N, N))
        for i in range(N):
            for j in range(i + 1):
                A[i][j] = A[j][i]
            else:
                A[i][j + 1::] = torch.randint(6, (1, N - i - 1))
        yield A


def get_labeled_edges_as_tuple(g: torch.Tensor) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], str]]:
    tuples: List[Tuple[int, int]] = []
    labels: Dict[Tuple[int, int], str] = {}
    for i in range(g.shape[0]):
        for j in range(i, g.shape[0]):
            if g[i][j] != 0:
                tuples.extend([(i + 1, j + 1), (j + 1, i + 1)])
                labels[(i + 1, j + 1)] = str(g[i][j].item())
    return tuples, labels


def main():
    for graph in generate_graphs(1, 5):
        print(graph)
        pg = nx.Graph()
        tuples, labels = get_labeled_edges_as_tuple(graph)
        pg.add_edges_from(tuples)
        pos = nx.spring_layout(pg)
        nx.draw_networkx_edge_labels(pg, pos, labels)
        nx.draw_networkx_nodes(pg, pos)
        nx.draw_networkx_edges(pg, pos)
        nx.draw_networkx_labels(pg, pos)
        plt.show()


if __name__ == "__main__":
    main()