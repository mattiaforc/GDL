from torch import optim
from G2G.preprocess.generate import generate_dataset
from G2G.utils import shortest_path_length, adj_to_shortest_path, get_ap_score, get_acc
from G2G.model.model import GAE
import matplotlib.pyplot as plt
import networkx as nx
import tqdm
import torch
import numpy as np


def test():
    x, y = generate_dataset(1, 10)
    print("Shortest path length:\t", shortest_path_length(y[x[0]][(1, 6)]))
    try:
        assert shortest_path_length(y[x[0]][(1, 6)]) == len(nx.shortest_path(x[0].graph, 1, 6, weight="weight")) - 1
        assert adj_to_shortest_path(y[x[0]][(1, 6)], 1) == nx.shortest_path(x[0].graph, 1, 6, weight="weight")[1:]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        assert shortest_path_length(y[x[0]][(1, 6)]) == 0

    gae: GAE = GAE(10, 10, 10)
    optimizer = optim.Adam(gae.parameters(), lr=0.05)
    # TODO: switch numpy to torch
    loss_history = np.zeros(1000)
    norm = y[x[0]][(1, 6)].shape[0] * y[x[0]][(1, 6)].shape[0] / float(
        (y[x[0]][(1, 6)].shape[0] * y[x[0]][(1, 6)].shape[0] - y[x[0]][(1, 6)].sum()) * 2)
    weights = torch.mean(y[x[0]][(1, 6)], dim=0) + 1e-4
    print("Calculated loss weights for class:\t", weights)

    for epoch in tqdm.trange(1000):
        optimizer.zero_grad()
        A_hat = gae(y[x[0]][(1, 6)])
        loss = gae.loss(A_hat, torch.max(y[x[0]][(1, 6)], 1)[1], norm, weights)
        loss.backward()
        optimizer.step()
        loss_history[epoch] = loss.detach().numpy()

    A_mod_hat = torch.stack([torch.where(x == torch.max(x), torch.max(x), torch.tensor(0.0)) for x in A_hat])
    plt.plot(loss_history)
    plt.show()
    print("Initial loss:\t", loss_history[0], "\tFinal loss:\t", loss_history[-1])
    print("Original matrix:\n", y[x[0]][(1, 6)], "\nReconstructed matrix:\n", A_hat.data, "\nModified matrix:\n",
          A_mod_hat)
    """
    for graph in generate_graphs(1, 7):
        graph.print()
        s = GraphWrapper(shortest_path_as_adjacency_matrix(graph, 1, 6), pos=graph.pos)
        print(nx.shortest_path(graph.graph, 1, 6, weight="weight"))
        s.print()
    """


if __name__ == "__main__":
    test()
