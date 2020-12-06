import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import argparse
import pathlib as path


layouts = {
    "circular": nx.circular_layout,
    "kamada_kawai": nx.kamada_kawai_layout,
    "random": nx.random_layout,
    "shell": nx.shell_layout,
    "spring": nx.spring_layout,
    "spectral": nx.spectral_layout,
    "fruchterman_reingold": nx.fruchterman_reingold_layout,
    "spiral": nx.spiral_layout}


def graph_plotter(A: np.ndarray, **draw_args):
    g = nx.convert_matrix.from_numpy_matrix(A)
    nx.draw(g, node_size=50, **draw_args)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickled_files', type=str, default=None,
                        help='pems_bay or metr_la')
    parser.add_argument('--seattle_data', type=bool, action='store_true',
                        help='to use Seattle data')
    parser.add_argument('--layout', type=str, default=None,
                        help='graph layout: circular, amada_kawai, random_layout,'
                        'shell, spring, spectral, fruchterman_reingold')
    args = parser.parse_args()

    if args.pickled_files:
        place = args.pickled_files
        place_path = path.Path("../data")/place
        with open(place_path/f"adj_mx_{place.split('_')[1]}.pkl", "rb") as f:
            sensor_ids, sensor_id_to_ind, A = pickle.load(f, encoding='latin-1')
        pos = np.load(place_path/f'pos_{place.split("_")[1]}.npy')
        graph_plotter(A, pos=pos)

    elif args.seattle_data:
        A = np.load("Seattle_Loop_Dataset/Loop_Seattle_2015_A.npy")
        mp = pd.read_csv("Seattle_Loop_Dataset/nodes_loop_mp_list.csv")["milepost"]
        mp = list(map(lambda entry: entry[1:4], mp))
        col = list(map(lambda entry: 'purple' if entry=='090' else 'y' if entry=='405' else 'b' if entry=='520' else 'r', mp))
        graph_plotter(A, node_color=col, layout=layouts[args.layout])

