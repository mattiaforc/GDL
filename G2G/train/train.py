import torch
from typing import List, Dict, Tuple, Iterable
from torch import optim
from tqdm import tqdm, trange
from G2G.model.graph_wrapper import GraphWrapper
from G2G.model.model import Predictor
from G2G.utils import reconstructed_matrix_to_shortest_path, adj_to_shortest_path, get_combo, prepare_input


def train(x: List[GraphWrapper], y: Dict[GraphWrapper, Dict[Tuple[int, int], torch.Tensor]], iterations: int, lr: float,
          tqdm_enabled: bool = True) -> Tuple[Predictor, float, torch.Tensor]:
    assert x != []
    predictor: Predictor = Predictor(*x[0].adj.shape)
    optimizer = optim.Adam(predictor.parameters(), lr=lr)
    custom_range: Iterable = trange(iterations) if tqdm_enabled else range(iterations)
    loss_history = torch.zeros(iterations)

    for epoch in custom_range:
        combo: List[Tuple[int, int]] = get_combo(x[0].adj.shape[0], len(x))
        for graph, c in zip(x, combo):
            optimizer.zero_grad()
            A_hat = predictor(prepare_input(c[0], c[1], graph.adj.shape[0]), graph.adj)
            loss = predictor.loss(A_hat, y[graph][(c[0], c[1])])
            loss.backward()
            optimizer.step()
            loss_history[epoch] += loss.detach().item()

    combo: List[Tuple[int, int]] = get_combo(x[0].adj.shape[0], len(x))
    a = [
        reconstructed_matrix_to_shortest_path(predictor(prepare_input(c[0], c[1], x[0].adj.shape[0]), g.adj).data, c[0],
                                              c[1]) == adj_to_shortest_path(
            y[g][(c[0], c[1])], c[0]) for g, c in zip(x, combo)]

    accuracy = sum(a) / len(a) * 100
    # print("Number of graphs: ", len(x), "\tDimension of each graph: ", x[0].adj.shape[0], "\tAccuracy: ", accuracy,
    #       "%")
    return predictor, accuracy, loss_history
