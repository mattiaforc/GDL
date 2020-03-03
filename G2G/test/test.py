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


# 10 iters
# training: {'total': 36.30222222222223, 'long': 16.626752243190598, 'short': 81.33450266754366, 'no_path': 4.575555555555556,
#  'invalid_path': 33.973333333333336}
# validation: {'total': 34.42222222222222, 'long': 14.689932372081154, 'short': 79.73190741283328, 'no_path': 4.973333333333333,
#  'invalid_path': 34.471111111111114}

# 15 iters
# training: {'total': 39.00222222222222, 'long': 20.219050356036657, 'short': 81.99225316085654, 'no_path': 4.8933333333333335, 'invalid_path': 30.591111111111115}
# validation:  {'total': 36.29777777777778, 'long': 17.254689294372845, 'short': 80.02490477585702, 'no_path': 5.611111111111111, 'invalid_path': 31.32888888888889}

# 40? iters
# training:  {'total': 44.18222222222222, 'long': 26.52233611137721, 'short': 84.60133011766425, 'no_path': 5.155555555555556, 'invalid_path': 26.531111111111112}
# validation:  {'total': 40.17777777777778, 'long': 22.046063544723747, 'short': 81.8121886903018, 'no_path': 6.202222222222222, 'invalid_path': 27.58}

# 5 iters 10000 graph
# training: {'total': 40.99933333333333, 'long': 21.07801644973249, 'short': 86.54884060617127, 'no_path': 2.830888888888889, 'invalid_path': 30.356666666666666}
# validation: {'total': 40.68266666666667, 'long': 20.87399561374106, 'short': 86.05839896452625, 'no_path': 2.915111111111111, 'invalid_path': 30.369777777777777}

# 10 iters 10000 graph
# Training: {'total': 44.13133333333333, 'long': 23.533977481434164, 'short': 91.22658389629359, 'no_path': 2.2093333333333334, 'invalid_path': 28.154444444444444}
# Validation: {'total': 43.67444444444445, 'long': 23.115648473588273, 'short': 90.76848825219929, 'no_path': 2.303777777777778, 'invalid_path': 28.203111111111113}

def ray_schedule():
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

    max_iter = 50
    search_space = {
        "lr": 0.001,
        "max_iter": max_iter,
        "gn": gn,
        "dim": dim,
        "hidden": tune.sample_from(lambda _: np.random.randint(dim, dim * 3)),
        "k": tune.sample_from(lambda _: np.random.choice((dim - 2) // 2 if dim <= 10 else dim // 4) + 2),
        "dropout": tune.choice([0.2, 0.3, 0.4])
    }

    analysis = tune.run(train_tune, num_samples=50,
                        scheduler=ASHAScheduler(metric="mean_accuracy", mode="max", grace_period=1),
                        config=search_space, verbose=2)

    print("Best config is", analysis.get_best_config(metric="mean_accuracy"))


@timer
def schedule():
    gn = 10000
    dim = 10
    if not os.path.isfile(f"../dataset/x-gn:{gn}-dim:{dim}-dataset.pt") \
            or not os.path.isfile(f"../dataset/y-gn:{gn}-dim:{dim}-dataset.pt") \
            or not os.path.isfile(f"../dataset/x-val-gn:{gn}-dim:{dim}-dataset.pt") \
            or not os.path.isfile(f"../dataset/y-val-gn:{gn}-dim:{dim}-dataset.pt"):
        save_on_hdd(*generate_dataset(gn, dim), path=f"../dataset/", name=f"gn:{gn}-dim:{dim}-dataset")
        save_on_hdd(*generate_dataset(gn, dim), path=f"../dataset/", name=f"val-gn:{gn}-dim:{dim}-dataset")
    x = torch.load(f"../dataset/x-gn:{gn}-dim:{dim}-dataset.pt")
    y = torch.load(f"../dataset/y-gn:{gn}-dim:{dim}-dataset.pt")
    x_val = torch.load(f"../dataset/x-val-gn:{gn}-dim:{dim}-dataset.pt")
    y_val = torch.load(f"../dataset/y-val-gn:{gn}-dim:{dim}-dataset.pt")
    max_iter = 5
    hidden = 150
    k = 3
    dropout = 0.2
    predictor = Predictor(dim, dim, hidden=hidden, k=k, dropout=dropout)
    if os.path.isfile(f"../dataset/model-gn:{gn}-dim:{dim}-hidden:{hidden}-k:{k}.pt"):
        predictor.load_state_dict(torch.load(f"../dataset/model-gn:{gn}-dim:{dim}-hidden:{hidden}-k:{k}.pt"))

    predictor, loss_history, acc, val = train(predictor, x, y, {"lr": 0.0001, "iterations": max_iter}, checkpoint=0,
                                              validation_x=x_val, validation_y=y_val)
    plt.plot(loss_history)
    plt.show()
    torch.save(predictor.state_dict(), f"../dataset/model-gn:{gn}-dim:{dim}-hidden:{hidden}-k:{k}.pt")

    predictor.eval()
    print("Score on train set:")
    print(acc)
    # print(get_score(predictor, x, y))
    print("Score on validation set:")
    print(val)
    # print(get_score(predictor, x_val, y_val))

    return 0


if __name__ == "__main__":
    torch.manual_seed(0)
    schedule()
    # ray_schedule()
