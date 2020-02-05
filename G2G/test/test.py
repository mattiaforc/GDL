from G2G.preprocess.generate import generate_dataset, GraphWrapper
from G2G.utils import shortest_path_length, adj_to_shortest_path, reconstructed_matrix_to_shortest_path
from G2G.model.model import Predictor
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch


def test(start: int, end: int, graph_number: int = 1, dim: int = 10):
    x, y = generate_dataset(graph_number, dim)
    print("Shortest path length:\t", shortest_path_length(y[x[0]][(start, end)]))
    try:
        assert shortest_path_length(y[x[0]][(start, end)]) == len(
            nx.shortest_path(x[0].graph, start, end, weight="weight")) - 1
        assert adj_to_shortest_path(y[x[0]][(start, end)], start) == nx.shortest_path(x[0].graph, start, end,
                                                                                      weight="weight")[1:]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        assert shortest_path_length(y[x[0]][(start, end)]) == 0

    predictor: Predictor = Predictor(dim, dim)
    optimizer = optim.Adam(predictor.parameters(), lr=0.01)
    loss_history = np.zeros(200)

    for epoch in tqdm(range(200)):
        optimizer.zero_grad()
        A_hat = predictor(torch.eye(*x[0].adj.shape), x[0].adj)
        loss = predictor.loss(A_hat, y[x[0]][(start, end)])
        loss.backward()
        optimizer.step()
        loss_history[epoch] = loss.detach().numpy()

    plt.plot(loss_history)
    plt.show()
    print("Initial loss:\t", loss_history[0], "\tFinal loss:\t", loss_history[-1])
    print("\nAdjacency matrix of graph:\n", x[0].adj)
    print("\nShortest nx-calculated matrix:\n", y[x[0]][(start, end)], "\nReconstructed matrix:\n", A_hat.data)
    print("\nShortest path (output of the net): \t", reconstructed_matrix_to_shortest_path(A_hat.data, start, end))
    s = GraphWrapper(A_hat.data, pos=x[0].pos)
    print("Shortest nx path:\t", nx.shortest_path(s.graph, start, end, weight="weight"))

    """
    for graph in generate_graphs(1, 7):
        graph.print()
        s = GraphWrapper(shortest_path_as_adjacency_matrix(graph, 1, 6), pos=graph.pos)
        print(nx.shortest_path(graph.graph, 1, 6, weight="weight"))
        s.print()
    """


if __name__ == "__main__":
    test(2, 7, graph_number=1, dim=10)
