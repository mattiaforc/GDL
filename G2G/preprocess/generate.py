import torch
import itertools
from typing import List, Tuple, Dict, Generator
from tqdm import tqdm
from G2G.model.graph_wrapper import GraphWrapper
from G2G.utils import shortest_as_adj_from_graph_wrapper


def generate_graphs(iterations: int, N: int, random: str = 'randn') -> Generator[GraphWrapper, None, None]:
    for _ in range(iterations):
        A = torch.zeros((N, N))
        for i in range(N):
            for j in range(i + 1):
                A[i][j] = A[j][i]
            else:
                A[i][j + 1::] = torch.nn.functional.relu(
                    torch.randn((1, N - i - 1)) if random == 'randn' else torch.randint(0, 10, (1, N - i - 1)))
        yield GraphWrapper(A)


def generate_dataset(iterations: int, N: int, random: str = 'randn') \
        -> Tuple[List[GraphWrapper], Dict[GraphWrapper, Dict[Tuple[int, int], torch.Tensor]]]:
    y = {}
    x = []
    for graph in tqdm(generate_graphs(iterations, N, random=random), total=iterations):
        x.append(graph)
        y[graph] = {}
        for combo in itertools.combinations(range(N), r=2):
            y[graph][combo] = shortest_as_adj_from_graph_wrapper(graph, *combo)
    return x, y
