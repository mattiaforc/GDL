from tqdm import tqdm, trange
from G2G.preprocess.generate import generate_dataset
from G2G.train.train import train
from G2G.utils import shortest_path_length, adj_to_shortest_path
import networkx as nx
import torch
import matplotlib.pyplot as plt


def find_best_dataset(limit: int = 100, graph_number: int = 100, dim: int = 10,
                      iterations: int = 500, write_hdd: bool = True):
    cached_max = 0.

    for _ in trange(limit):
        x, y = generate_dataset(graph_number, dim, tqdm_enabled=False)
        predictor, accuracy, loss_history = train(x, y, iterations, lr=0.01, tqdm_enabled=False)

        if accuracy > cached_max:
            plt.plot(loss_history)
            plt.show()
            tqdm.write("\tNew accuracy: {}".format(accuracy))
            if write_hdd:
                with open("../dataset/gn:{}-dim:{}-iter:{}-dataset-x.pt".format(graph_number, dim, iterations),
                          mode='wb') as output:
                    torch.save(x, output)
                with open("../dataset/gn:{}-dim:{}-iter:{}-dataset-y.pt".format(graph_number, dim, iterations),
                          mode='wb') as output:
                    torch.save(y, output)
                torch.save(predictor.state_dict(),
                           "../dataset/gn:{}-dim:{}-iter:{}-model.pt".format(graph_number, dim, iterations))
            cached_max = accuracy


def test(start: int, end: int, graph_number: int = 100, dim: int = 10, iterations: int = 500):
    x, y = generate_dataset(graph_number, dim)

    try:
        assert shortest_path_length(y[x[0]][(start, end)]) == len(
            nx.shortest_path(x[0].graph, start, end, weight="weight")) - 1
        assert adj_to_shortest_path(y[x[0]][(start, end)], start) == nx.shortest_path(x[0].graph, start, end,
                                                                                      weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        assert shortest_path_length(y[x[0]][(start, end)]) == 0

    """
    # plt.plot(loss_history)
    # plt.show()
    x[0].print()
    print("Initial loss:\t", loss_history[0], "\tFinal loss:\t", loss_history[-1])
    print("\nAdjacency matrix of graph:\n", x[0].adj)
    print("\nShortest nx-calculated matrix:\n", y[x[0]][(start, end)], "\nReconstructed matrix:\n", A_hat.data)
    print("\nShortest path (output of the net): \t", reconstructed_matrix_to_shortest_path(A_hat.data, start, end))
    s = GraphWrapper(y[x[0]][(start, end)], pos=x[0].pos)
    GraphWrapper(shortest_path_to_adj(reconstructed_matrix_to_shortest_path(A_hat.data, start, end), dim),
                 pos=x[0].pos).print()
    print("Shortest nx path:\t", nx.shortest_path(s.graph, start, end, weight="weight"))
    
    # -------------------------------------------------------------------
    for graph in generate_graphs(1, 7):
        graph.print()
        s = GraphWrapper(shortest_path_as_adjacency_matrix(graph, 1, 6), pos=graph.pos)
        print(nx.shortest_path(graph.graph, 1, 6, weight="weight"))
        s.print()
    """


if __name__ == "__main__":
    find_best_dataset(graph_number=100, dim=10, iterations=500, write_hdd=True)
