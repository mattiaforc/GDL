from G2G.preprocess.generate import generate_dataset
from G2G.utils import shortest_path_length, adj_to_shortest_path, reconstructed_matrix_to_shortest_path
from G2G.model.model import Predictor
from torch import optim
from tqdm import tqdm
import networkx as nx


def test(start: int, end: int, graph_number: int = 1, dim: int = 10):
    x, y = generate_dataset(graph_number, dim)
    predictor: Predictor = Predictor(dim, dim)
    optimizer = optim.Adam(predictor.parameters(), lr=0.001)

    try:
        assert shortest_path_length(y[x[0]][(start, end)]) == len(
            nx.shortest_path(x[0].graph, start, end, weight="weight")) - 1
        assert adj_to_shortest_path(y[x[0]][(start, end)], start) == nx.shortest_path(x[0].graph, start, end,
                                                                                      weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        assert shortest_path_length(y[x[0]][(start, end)]) == 0

    # loss_history = np.zeros(200)

    for _ in tqdm(range(500)):
        for graph in x:
            optimizer.zero_grad()
            A_hat = predictor(graph.adj)
            loss = predictor.loss(A_hat, y[graph][(start, end)])
            loss.backward()
            optimizer.step()
            # loss_history[epoch] = loss.detach().numpy()

    a = [reconstructed_matrix_to_shortest_path(predictor(g.adj).data, start, end) == adj_to_shortest_path(
        y[g][(start, end)], start) for g in x]
    # a = [reconstructed_matrix_to_shortest_path(rec_adj, start, end) == label for rec_adj, label in
    #      map(lambda g: (predictor(g.adj).data, y[g][(start, end)]), x)]

    accuracy = sum(a) / len(a) * 100
    print("Number of graphs: ", graph_number, "\tDimension of each graph: ", dim, "\tAccuracy: ", accuracy, "%")

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
    test(1, 47, graph_number=100, dim=50)
