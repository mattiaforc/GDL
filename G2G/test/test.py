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
    print(shortest_path_length(y[x[0]][(1, 6)]))
    try:
        assert shortest_path_length(y[x[0]][(1, 6)]) == len(nx.shortest_path(x[0].graph, 1, 6, weight="weight")) - 1
        assert adj_to_shortest_path(y[x[0]][(1, 6)], 1) == nx.shortest_path(x[0].graph, 1, 6, weight="weight")[1:]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        assert shortest_path_length(y[x[0]][(1, 6)]) == 0

    print(y[x[0]][(1, 6)])

    gae: GAE = GAE(10, 10, 10)
    optimizer = optim.Adam(gae.parameters(), lr=0.01)
    # TODO: switch numpy to torch
    loss_history = np.zeros(1000)
    norm = y[x[0]][(1, 6)].shape[0] * y[x[0]][(1, 6)].shape[0] / float(
        (y[x[0]][(1, 6)].shape[0] * y[x[0]][(1, 6)].shape[0] - y[x[0]][(1, 6)].sum()) * 2)
    weights = torch.mean(y[x[0]][(1, 6)], dim=0)
    print(weights)

    for epoch in tqdm.trange(1000):
        optimizer.zero_grad()
        A_hat = gae(y[x[0]][(1, 6)])
        loss = gae.loss(A_hat, torch.max(y[x[0]][(1, 6)], 1)[1], norm, weights)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print("Accuracy: " + str(get_acc(A_hat, y[x[0]][(1, 6)])) +
                  "\tAP score: " + str(get_ap_score(A_hat, y[x[0]][(1, 6)]))
                  )

        loss_history[epoch] = loss.detach().numpy()

    plt.plot(loss_history)
    plt.show()
    print(loss_history[0])
    print(loss_history[len(loss_history) - 1])
    print(y[x[0]][(1, 6)], A_hat.data, sep='\n')
    """
    for graph in generate_graphs(1, 7):
        graph.print()
        s = GraphWrapper(shortest_path_as_adjacency_matrix(graph, 1, 6), pos=graph.pos)
        print(nx.shortest_path(graph.graph, 1, 6, weight="weight"))
        s.print()
    """


if __name__ == "__main__":
    test()
