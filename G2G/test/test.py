from torch import optim
from G2G.preprocess.generate import generate_dataset
from G2G.utils import shortest_path_length, adj_to_shortest_path
from G2G.model.model import Predictor
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import torch
import numpy as np

from typing import List


def test():
    x, y = generate_dataset(1, 10)
    print("Shortest path length:\t", shortest_path_length(y[x[0]][(1, 6)]))
    try:
        assert shortest_path_length(y[x[0]][(1, 6)]) == len(nx.shortest_path(x[0].graph, 1, 6, weight="weight")) - 1
        assert adj_to_shortest_path(y[x[0]][(1, 6)], 1) == nx.shortest_path(x[0].graph, 1, 6, weight="weight")[1:]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        assert shortest_path_length(y[x[0]][(1, 6)]) == 0

    predictor: Predictor = Predictor(10, 10)
    optimizer = optim.Adam(predictor.parameters(), lr=0.01)
    loss_history = np.zeros(200)

    for epoch in tqdm(range(200)):
        optimizer.zero_grad()
        A_hat = predictor(torch.ones(*x[0].adj.shape), x[0].adj)
        loss = predictor.loss(A_hat, y[x[0]][(1, 6)])
        loss.backward()
        optimizer.step()
        loss_history[epoch] = loss.detach().numpy()

    plt.plot(loss_history)
    plt.show()
    print("Initial loss:\t", loss_history[0], "\tFinal loss:\t", loss_history[-1])
    print("\nAdjacency matrix of graph:\n", x[0].adj)
    print("\nShortest nx-calculated matrix:\n", y[x[0]][(1, 6)], "\nReconstructed matrix:\n", A_hat.data)

    """
    for graph in generate_graphs(1, 7):
        graph.print()
        s = GraphWrapper(shortest_path_as_adjacency_matrix(graph, 1, 6), pos=graph.pos)
        print(nx.shortest_path(graph.graph, 1, 6, weight="weight"))
        s.print()
    """


if __name__ == "__main__":
    test()
