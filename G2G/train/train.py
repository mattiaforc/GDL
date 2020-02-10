import torch
from typing import List, Dict, Tuple, Iterable
from torch import optim
from tqdm import tqdm
from G2G.model.graph_wrapper import GraphWrapper
from G2G.model.model import Predictor
from G2G.utils import reconstructed_matrix_to_shortest_path, adj_to_shortest_path


def train(x: List[GraphWrapper], y: Dict[GraphWrapper, Dict[Tuple[int, int], torch.Tensor]], iterations: int,
          start: int, end: int, lr: float, tqdm_enabled: bool = True) -> Tuple[Predictor, float]:
    assert x != []
    predictor: Predictor = Predictor(*x[0].adj.shape)
    optimizer = optim.Adam(predictor.parameters(), lr=lr)

    # loss_history = np.zeros(200)
    custom_range: Iterable = tqdm(range(iterations)) if tqdm_enabled else range(iterations)

    for _ in custom_range:
        for graph in x:
            optimizer.zero_grad()
            A_hat = predictor(graph.adj)
            loss = predictor.loss(A_hat, y[graph][(start, end)])
            loss.backward()
            optimizer.step()
            # loss_history[epoch] = loss.detach().numpy()

    a = [reconstructed_matrix_to_shortest_path(predictor(g.adj).data, start, end) == adj_to_shortest_path(
        y[g][(start, end)], start) for g in x]

    accuracy = sum(a) / len(a) * 100
    # print("Number of graphs: ", len(x), "\tDimension of each graph: ", x[0].adj.shape[0], "\tAccuracy: ", accuracy,
    #       "%")
    return predictor, accuracy
