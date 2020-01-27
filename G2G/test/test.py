from G2G.preprocess.generate import generate_dataset
from G2G.utils import shortest_path_length, adj_to_shortest_path
import networkx as nx


def test():
    x, y = generate_dataset(1, 7)
    print(shortest_path_length(y[x[0]][(1, 6)]))
    try:
        assert shortest_path_length(y[x[0]][(1, 6)]) == len(nx.shortest_path(x[0].graph, 1, 6, weight="weight")) - 1
        assert adj_to_shortest_path(y[x[0]][(1, 6)], 1) == nx.shortest_path(x[0].graph, 1, 6, weight="weight")[1:]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        assert shortest_path_length(y[x[0]][(1, 6)]) == 0

    """
    for graph in generate_graphs(1, 7):
        graph.print()
        s = GraphWrapper(shortest_path_as_adjacency_matrix(graph, 1, 6), pos=graph.pos)
        print(nx.shortest_path(graph.graph, 1, 6, weight="weight"))
        s.print()
    """


if __name__ == "__main__":
    test()
