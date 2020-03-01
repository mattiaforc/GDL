import torch
from typing import List, Dict, Tuple, Iterable
from ray import tune
from torch import optim
from tqdm import trange
from G2G.model.graph_wrapper import GraphWrapper
from G2G.model.model import Predictor
from G2G.preprocess.generate import generate_dataset
from G2G.utils import reconstructed_matrix_to_shortest_path, adj_to_shortest_path, get_all_combo, prepare_input, \
    get_score
from G2G.decorators.decorators import logger, Formatter, timer


def train_tune(config: Dict):
    x = torch.load("/home/malattia/Workspace/Tesi/G2G/dataset/gn:100-dim:10-iter:150-dataset-x.pt")
    y = torch.load("/home/malattia/Workspace/Tesi/G2G/dataset/gn:100-dim:10-iter:150-dataset-y.pt")
    return train(predictor, x, y, tqdm_enabled=False, config=config, tune_on=True)


@logger(Formatter(lambda x: "Training results:\nAccuracy: " + str(x[1]) + "\nLast loss: " + str(x[2][-1].item())))
@timer
def train(predictor: Predictor, x: List[GraphWrapper], y: Dict[str, Dict[Tuple[int, int], torch.Tensor]], config: Dict,
          tqdm_enabled: bool = True, tune_on: bool = False) -> Tuple[Predictor, float, torch.Tensor]:
    # config = {iterations: int, lr: float, combo_num: int}
    assert x != []
    # predictor: Predictor = Predictor(*x[0].laplacian.shape)
    optimizer = optim.Adam(predictor.parameters(), lr=config["lr"])
    custom_range: Iterable = trange(config["iterations"]) if tqdm_enabled else range(config["iterations"])
    loss_history = torch.zeros(config["iterations"])

    dim: int = x[0].laplacian.shape[0]
    combo: List[Tuple[int, int]] = get_all_combo(dim)
    for epoch in custom_range:
        for graph in x:
            for c in get_all_combo(dim):
                optimizer.zero_grad()
                A_hat = predictor(prepare_input(c[0], c[1], dim, graph.laplacian), graph.laplacian)
                loss = predictor.loss(A_hat, y[str(graph)][(c[0], c[1])])
                loss.backward()
                optimizer.step()
                loss_history[epoch] += loss.detach().item()

    a = []
    for g in x:
        for c in combo:
            a.append(reconstructed_matrix_to_shortest_path(
                predictor(prepare_input(c[0], c[1], dim, g.laplacian), g.laplacian).data, c[0],
                c[1]) == adj_to_shortest_path(y[str(g)][(c[0], c[1])], c[0]))
    accuracy = sum(a) / len(a) * 100
    if tune_on: tune.track.log(mean_accuracy=accuracy)

    return predictor, accuracy, loss_history


def train_batch(predictor, num_batch: int, graph_per_batch: int, graph_dim: int, iterations: int,
                validation_x: List[GraphWrapper], validation_y: Dict[str, Dict[Tuple[int, int], torch.Tensor]],
                lr: float = 0.001, tqdm_enabled: bool = True):
    optimizer = optim.Adam(predictor.parameters(), lr=lr)
    batch_range: Iterable = trange(num_batch if tqdm_enabled else range(num_batch))
    custom_range: Iterable = range(iterations)
    loss_history = torch.zeros(iterations)

    for batch_num in batch_range:
        batch_x, batch_y = generate_dataset(graph_per_batch * num_batch, graph_dim, tqdm_enabled=False)
        for epoch in custom_range:
            for graph in batch_x[graph_per_batch * batch_num:graph_per_batch * (batch_num + 1):]:
                for c in get_all_combo(graph_dim):
                    optimizer.zero_grad()
                    A_hat = predictor(prepare_input(c[0], c[1], graph_dim, graph.laplacian), graph.laplacian)
                    loss = predictor.loss(A_hat, batch_y[str(graph)][(c[0], c[1])])
                    loss.backward()
                    optimizer.step()
                    loss_history[epoch] += loss.detach().item()

        print("Score on batch-trained dataset:")
        print(get_score(predictor, batch_x, batch_y))
        print("Score on validation dataset:")
        print(get_score(predictor, validation_x, validation_y))

    loss_history /= graph_per_batch * num_batch
    return predictor, loss_history
