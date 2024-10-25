from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import TypeAlias, Literal, TypeVar
    N: TypeAlias = TypeVar['N']
    AdjMatrix: TypeAlias = np.ndarray[tuple[N, N], int]
    Layout: TypeAlias = np.ndarray[tuple[N, Literal[2]], int]

from pathlib import Path
import csv
from tulip import tlp

from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import shortest_path

import numpy as np

from tqdm import tqdm


DATA_DIR = Path('data')


def load_tulip_graph(file_path: str) -> tlp.Graph:
    g = tlp.loadGraph(file_path)
    if not tlp.ConnectedTest.isConnected(g):
        params = tlp.getDefaultPluginParameters('Make Connected', g)
        success = g.applyAlgorithm('Make Connected', params)
    return g


def tulip_graph_to_adj_matrix(graph: tlp.Graph, n_nodes: int=-1) -> AdjMatrix:
    n_nodes = max(graph.numberOfNodes(), n_nodes)
    adj_mat = np.zeros((n_nodes, n_nodes), dtype=int)
    for node in graph.getNodes():
        for neighbor in graph.getInOutNodes(node):
            adj_mat[node.id, neighbor.id] = 1
    return adj_mat


def compute_stress(adj_mat: AdjMatrix, layout: Layout) -> float:
    ideal_dist = shortest_path(adj_mat, directed=False)
    layout_dist = squareform(pdist(layout))
    finite_mask = np.isfinite(ideal_dist)
    return np.sum((ideal_dist[finite_mask] - layout_dist[finite_mask])**2)


def build_stress_predictor_dataset(dataset_root_dir: Path, tlp_graph_dir: Path, n_shuffle: int) -> None:
    adj_matrices_dir = dataset_root_dir.joinpath('adj_matrices')
    layout_dir = dataset_root_dir.joinpath('layouts')
    stress_values_file = dataset_root_dir.joinpath('stress_values.csv')

    adj_matrices_dir.mkdir(exist_ok=True)
    layout_dir.mkdir(exist_ok=True)

    with stress_values_file.open(mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        for tlp_graph_file in tqdm(tlp_graph_dir.glob('*.tlpb.gz')):
            graph = load_tulip_graph(str(tlp_graph_file))
            adj_mat = tulip_graph_to_adj_matrix(graph)

            graph_name = tlp_graph_file.stem
            np.save(adj_matrices_dir.joinpath(f"{graph_name}.npy"), adj_mat)

            for i in range(n_shuffle):
                layout = np.random.rand(graph.numberOfNodes(), 2)
                layout_name = f"{graph_name}_layout_{i}"
                np.save(layout_dir.joinpath(f"{layout_name}.npy"), layout)

                stress = compute_stress(adj_mat, layout)
                writer.writerow((graph_name, layout_name, stress))


class StressPredictorDataset:
    def __init__(self, root_dir: Path) -> None:
        self.adj_matrices_dir = root_dir.joinpath('adj_matrices')
        self.layouts_dir = root_dir.joinpath('layouts')

        self.adj_mat_names = []
        self.layout_names = []
        self.stress_values = []
        with root_dir.joinpath('stress_values.csv').open(mode='r') as csv_file:
            reader = csv.reader(csv_file)
            for graph_name, layout_name, stress in reader:
                self.adj_mat_names.append(f"{graph_name}.npy")
                self.layout_names.append(f"{layout_name}.npy")
                self.stress_values.append(float(stress))

    def __getitem__(self, index: int) -> tuple[AdjMatrix, Layout, int]:
        adj_mat = np.load(self.adj_matrices_dir.joinpath(self.adj_mat_names[index]))
        layout = np.load(self.layouts_dir.joinpath(self.layout_names[index]))
        stress = self.stress_values[index]

        assert adj_mat.ndim == 2 and adj_mat.shape[0] == adj_mat.shape[1]
        assert layout.ndim == 2 and layout.shape[0] == adj_mat.shape[0] and layout.shape[1] == 2

        return adj_mat, layout, stress

    def __len__(self) -> int:
        return len(self.stress_values)


if __name__ == '__main__':
    import graph_processing as gp
    import matplotlib.pyplot as plt

    # build_stress_predictor_dataset(
    #     dataset_root_dir=Path('data', 'stress_predictor_dataset'),
    #     tlp_graph_dir=Path('data', 'rome_graphs'),
    #     n_shuffle=10
    # )

    dataset = StressPredictorDataset(Path('data', 'stress_predictor_dataset'))

    for i in np.random.choice(len(dataset), size=10, replace=False):
        adj_mat, layout, stress = dataset[i]

        fig, ax = plt.subplots()
        ax.set_title(f"stress value = {stress}")
        graph, _ = gp.AM2tlp(adj_mat)
        gp.writeTLPlayout(graph, "random_layout", layout)
        gp.draw_graph(graph, node_radiuses=0.1, ax=ax, layoutPropName="random_layout")
        plt.show()
