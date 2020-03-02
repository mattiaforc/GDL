from ray.tune.progress_reporter import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import os.path
import numpy as np
from G2G.decorators.decorators import logger, Formatter, timer
from G2G.model.model import Predictor
from G2G.preprocess.generate import generate_dataset
from G2G.train.train import train, train_tune
import torch
import matplotlib.pyplot as plt
from ray import tune
from G2G.utils import get_score, save_on_hdd


# Best config for 10 graphs 10 dim:
# {'lr': 0.001, 'max_iter': 100, 'gn': 10, 'dim': 10, 'hidden': 12, 'k': 2, 'dropout': 0.3}

@timer
def schedule():
    gn = 10
    dim = 10
    if not os.path.isfile(f"../dataset/x-gn:{gn}-dim:{dim}-dataset.pt") \
            or not os.path.isfile(f"../dataset/y-gn:{gn}-dim:{dim}-dataset.pt") \
            or not os.path.isfile(f"../dataset/x-val-gn:{gn}-dim:{dim}-dataset.pt") \
            or not os.path.isfile(f"../dataset/y-val-gn:{gn}-dim:{dim}-dataset.pt"):
        save_on_hdd(*generate_dataset(gn, dim), path=f"../dataset/", name=f"gn:{gn}-dim:{dim}-dataset")
        save_on_hdd(*generate_dataset(gn, dim), path=f"../dataset/", name=f"val-gn:{gn}-dim:{dim}-dataset")
    x = torch.load(f"../dataset/x-gn:{gn}-dim:{dim}-dataset.pt")
    y = torch.load(f"../dataset/y-gn:{gn}-dim:{dim}-dataset.pt")
    max_iter = 50
    hidden = 10
    k = 2
    predictor = Predictor(dim, dim, hidden=hidden, k=k, dropout=0.1)
    if os.path.isfile(f"../dataset/model-gn:{gn}-dim:{dim}-hidden:{hidden}-k:{k}.pt"):
        predictor.load_state_dict(torch.load(f"../dataset/model-gn:{gn}-dim:{dim}-hidden:{hidden}-k:{k}.pt"))
    predictor, loss_history, acc, val = train(predictor, x, y, {"lr": 0.001, "iterations": max_iter})
    plt.plot(loss_history)
    plt.show()
    torch.save(predictor.state_dict(), f"../dataset/model-gn:{gn}-dim:{dim}-hidden:{hidden}-k:{k}.pt")

    predictor.eval()
    print(get_score(predictor, x, y))
    print(get_score(predictor, *generate_dataset(gn, 10)))
    return 0


if __name__ == "__main__":
    torch.manual_seed(0)
    gn = 10
    dim = 10

    if not os.path.isfile(f"/home/malattia/Workspace/Tesi/G2G/dataset/gn:{gn}-dim:{dim}-dataset-x") \
            or not os.path.isfile(f"/home/malattia/Workspace/Tesi/G2G/dataset/gn:{gn}-dim:{dim}-dataset-y") \
            or not os.path.isfile(f"/home/malattia/Workspace/Tesi/G2G/dataset/val-gn:{gn}-dim:{dim}-dataset-x") \
            or not os.path.isfile(f"/home/malattia/Workspace/Tesi/G2G/dataset/val-gn:{gn}-dim:{dim}-dataset-y"):
        save_on_hdd(*generate_dataset(gn, dim),
                    path=f"/home/malattia/Workspace/Tesi/G2G/dataset/", name=f"gn:{gn}-dim:{dim}-dataset")
        save_on_hdd(*generate_dataset(gn, dim),
                    path=f"/home/malattia/Workspace/Tesi/G2G/dataset/", name=f"val-gn:{gn}-dim:{dim}-dataset")

    max_iter = 100
    search_space = {
        "lr": 0.001,
        "max_iter": max_iter,
        "gn": gn,
        "dim": dim,
        "hidden": tune.sample_from(lambda _: np.random.randint(dim, dim * 3)),
        "k": tune.sample_from(lambda _: np.random.choice((dim - 2) // 2 if dim <= 10 else dim // 4) + 1),
        "dropout": tune.choice([0.15, 0.2, 0.25, 0.3, 0.4])
    }

    analysis = tune.run(train_tune, num_samples=100,
                        scheduler=ASHAScheduler(metric="mean_accuracy", mode="max", grace_period=1),
                        config=search_space, verbose=2)

    print("Best config is", analysis.get_best_config(metric="mean_accuracy"))
    # schedule()
