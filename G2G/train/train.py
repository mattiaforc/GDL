import torch
from typing import List, Dict, Tuple, Iterable
from ray import tune
from torch import optim
from tqdm import trange
from G2G.model.graph_wrapper import GraphWrapper
from G2G.model.model import Predictor
from G2G.preprocess.generate import generate_dataset
from G2G.utils import get_all_combo, prepare_input, get_score, save_on_hdd
from G2G.decorators.decorators import logger, Formatter, timer
import os


def train_tune(config: Dict):
    gn = config["gn"]
    dim = config["dim"]

    predictor = Predictor(dim, dim, config['hidden'], config['k'], config['dropout'])
    max_iter = config["max_iter"]

    x = torch.load(f"/home/malattia/Workspace/Tesi/G2G/dataset/x-gn:{gn}-dim:{dim}-dataset.pt")
    y = torch.load(f"/home/malattia/Workspace/Tesi/G2G/dataset/y-gn:{gn}-dim:{dim}-dataset.pt")
    x_val = torch.load(f"/home/malattia/Workspace/Tesi/G2G/dataset/x-val-gn:{gn}-dim:{dim}-dataset.pt")
    y_val = torch.load(f"/home/malattia/Workspace/Tesi/G2G/dataset/y-val-gn:{gn}-dim:{dim}-dataset.pt")
    lr = config["lr"]

    return train(predictor, x, y, {"lr": lr, "iterations": max_iter}, tqdm_enabled=False, tune_on=True,
                 validation_x=x_val, validation_y=y_val)


# @logger(Formatter(lambda x: "Training results:\nAccuracy: " + str(x[1]) + "\nLast loss: " + str(x[2][-1].item())))
@timer
def train(predictor: Predictor, x: List[GraphWrapper], y: Dict[str, Dict[Tuple[int, int], torch.Tensor]], config: Dict,
          validation_x: List[GraphWrapper] = None, validation_y: Dict[str, Dict[Tuple[int, int], torch.Tensor]] = None,
          tqdm_enabled: bool = True, tune_on: bool = False) \
        -> Tuple[Predictor, torch.Tensor, Dict[str, float], Dict[str, float]]:
    # config = {iterations: int, lr: float}

    optimizer = optim.Adam(predictor.parameters(), lr=config["lr"])
    custom_range: Iterable = trange(config["iterations"]) if tqdm_enabled else range(config["iterations"])
    loss_history = torch.zeros(config["iterations"])
    dim: int = x[0].laplacian.shape[0]
    predictor.train()

    for epoch in custom_range:
        for graph in x:
            for c in get_all_combo(dim):
                optimizer.zero_grad()
                A_hat = predictor(prepare_input(c[0], c[1], dim, graph.laplacian), graph.laplacian)
                loss = predictor.loss(A_hat, y[str(graph)][(c[0], c[1])])
                loss.backward()
                optimizer.step()
                loss_history[epoch] += loss.detach().item()

    predictor.eval()
    val = get_score(predictor, validation_x,
                    validation_y) if validation_x is not None and validation_y is not None else None
    acc = get_score(predictor, x, y)
    if tune_on and validation_x is not None and validation_y is not None:
        tune.track.log(mean_accuracy=val['total'])
        torch.save(predictor.state_dict(),
                   f"/home/malattia/Workspace/Tesi/G2G/dataset/gn:{len(x)}-dim:{dim}-hidden:{predictor.GCN2.weight.shape[2]}-k:{predictor.GCN2.weight.shape[0]}-model.pt")
    return predictor, loss_history, acc, val
