import torch
import itertools
from typing import List, Tuple, Dict, Generator, Iterable
from tqdm import tqdm
from G2G.model.graph_wrapper import GraphWrapper
from G2G.utils import shortest_as_adj_from_graph_wrapper


def generate_graphs(iterations: int, N: int, device: torch.device, random: str = 'randn') -> Generator[
    GraphWrapper, None, None]:
    for _ in range(iterations):
        A = torch.zeros((N, N), device=device)
        for i in range(N):
            for j in range(i + 1):
                A[i][j] = A[j][i]
            else:
                A[i][j + 1::] = torch.nn.functional.relu(
                    torch.randn((1, N - i - 1), device=device) if random == 'randn' else torch.randint(0, 10,
                                                                                                       (1, N - i - 1),
                                                                                                       device=device))
        yield GraphWrapper(A, device=device)


def generate_dataset(iterations: int, N: int, device: torch.device, random: str = 'randn', tqdm_enabled: bool = True) \
        -> Tuple[List[GraphWrapper], Dict[str, Dict[Tuple[int, int], torch.Tensor]]]:
    y = {}
    x = []
    custom_range: Iterable = tqdm(generate_graphs(iterations, N, device=device, random=random), total=iterations) \
        if tqdm_enabled else generate_graphs(iterations, N, device=device, random=random)

    for graph in custom_range:
        x.append(graph)
        y[str(graph)] = {}
        for combo in itertools.combinations(range(1, N + 1), r=2):
            y[str(graph)][combo] = shortest_as_adj_from_graph_wrapper(graph, *combo, device=device)
    return x, y
