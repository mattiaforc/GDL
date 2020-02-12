import itertools

from ray.tune.schedulers import ASHAScheduler
from tqdm import tqdm, trange
import networkx as nx
from G2G.model.graph_wrapper import GraphWrapper
from G2G.model.model import Predictor
from G2G.preprocess.generate import generate_dataset
from G2G.train.train import train, train_tune
import torch
import matplotlib.pyplot as plt
from ray import tune

from G2G.utils import reconstructed_matrix_to_shortest_path, prepare_input, shortest_path_to_adj, adj_to_shortest_path


def find_best_dataset(limit: int = 100, graph_number: int = 100, dim: int = 10, iterations: int = 500, lr: float = 0.01,
                      write_hdd: bool = False) -> None:
    cached_max = 0.

    for _ in trange(limit):
        x, y = generate_dataset(graph_number, dim, tqdm_enabled=False)
        config = {"lr": lr, "iterations": iterations}
        predictor, accuracy, loss_history = train(x, y, config=config, tqdm_enabled=False)

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

    find_best_dataset(limit=10, graph_number=2, dim=10, iterations=300, lr=0.005, write_hdd=True)
    x = torch.load("../dataset/gn:2-dim:10-iter:300-dataset-x.pt")
    y = torch.load("../dataset/gn:2-dim:10-iter:300-dataset-y.pt")
    predictor: Predictor = Predictor(10, 10)
    predictor.load_state_dict(torch.load("../dataset/gn:2-dim:10-iter:300-model.pt"))

    while input("Stoppare? --stop") != "stop":
        g_num = int(input("Graph number: "))
        x[g_num - 1].print()
        print("\n",
              list(filter(
                  lambda combo: len(adj_to_shortest_path(y[str(x[g_num - 1])][(combo[0], combo[1])], combo[0])) > 2,
                  itertools.combinations(range(1, 10), r=2))),
              "\n")

        start = int(input("Start node: "))
        end = int(input("End node: "))
        guess = predictor(prepare_input(start, end, 10), x[g_num - 1].adj)
        print(guess.data)
        print("\nShortest path (output of the net): \t", reconstructed_matrix_to_shortest_path(guess.data, start, end))

        s = GraphWrapper(y[str(x[g_num - 1])][(start, end)], pos=x[g_num - 1].pos)
        GraphWrapper(shortest_path_to_adj(reconstructed_matrix_to_shortest_path(guess.data, start, end), 10),
                     pos=x[g_num - 1].pos).print()
        print("Shortest nx path:\t", nx.shortest_path(s.graph, start, end, weight="weight"))

    """
    search_space = {
        "lr": tune.loguniform(0.0001, 0.1),
        "iterations": tune.randint(100, 1000),
    }

    analysis = tune.run(train_tune, resources_per_trial={'gpu': 1}, num_samples=10,
                        scheduler=ASHAScheduler(metric="mean_accuracy", mode="max", grace_period=20),
                        config=search_space)
    """
