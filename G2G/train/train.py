import torch
from typing import List, Dict, Tuple, Iterable
from ray import tune
from torch import optim
from tqdm import trange
from G2G.model.graph_wrapper import GraphWrapper
from G2G.model.model import Predictor
from G2G.utils import reconstructed_matrix_to_shortest_path, adj_to_shortest_path, get_all_combo, prepare_input, \
    get_combo


def train_tune(config: Dict):
    x = torch.load("/home/malattia/Workspace/Tesi/G2G/dataset/gn:100-dim:10-iter:150-dataset-x.pt")
    y = torch.load("/home/malattia/Workspace/Tesi/G2G/dataset/gn:100-dim:10-iter:150-dataset-y.pt")
    return train(x, y, tqdm_enabled=False, config=config, tune_on=True)


def train(x: List[GraphWrapper], y: Dict[str, Dict[Tuple[int, int], torch.Tensor]], config: Dict,
          tqdm_enabled: bool = True, tune_on: bool = False) -> Tuple[Predictor, float, torch.Tensor]:
    # config = {iterations: int, lr: float, combo_num: int}

    assert x != []
    predictor: Predictor = Predictor(*x[0].adj.shape)
    optimizer = optim.Adam(predictor.parameters(), lr=config["lr"])
    custom_range: Iterable = trange(config["iterations"]) if tqdm_enabled else range(config["iterations"])
    loss_history = torch.zeros(config["iterations"])

    dim: int = x[0].adj.shape[0]
    combo: List[Tuple[int, int]] = get_all_combo(dim)
    for epoch in custom_range:
        for graph in x:
            for c in get_all_combo(dim):
                optimizer.zero_grad()
                A_hat = predictor(prepare_input(c[0], c[1], dim), graph.adj)
                loss = predictor.loss(A_hat, y[str(graph)][(c[0], c[1])])
                loss.backward()
                optimizer.step()
                loss_history[epoch] += loss.detach().item()

    a = []
    for g in x:
        for c in combo:
            a.append(reconstructed_matrix_to_shortest_path(predictor(prepare_input(c[0], c[1], dim), g.adj).data, c[0],
                                                           c[1]) == adj_to_shortest_path(y[str(g)][(c[0], c[1])], c[0]))
    accuracy = sum(a) / len(a) * 100
    if tune_on: tune.track.log(mean_accuracy=accuracy)

    return predictor, accuracy, loss_history
