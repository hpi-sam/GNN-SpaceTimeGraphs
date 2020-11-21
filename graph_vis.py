import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

A = np.load("Seattle_Loop_Dataset/Loop_Seattle_2015_A.npy")
X = np.load("Seattle_Loop_Dataset/Loop_Seattle_2015_A.npy")
mp = pd.read_csv("Seattle_Loop_Dataset/nodes_loop_mp_list.csv")["milepost"]
mp = list(map(lambda entry: entry[1:4], mp))
col = list(map(lambda entry: 'b' if entry=='090' else 'g' if entry=='405' else 'r' if entry=='520' else 'k', mp))
print(np.shape(A))
print(A)

all_layouts = [
    nx.circular_layout,
    nx.kamada_kawai_layout,
    nx.random_layout,
    nx.shell_layout,
    nx.spring_layout,
    nx.spectral_layout,
    nx.fruchterman_reingold_layout,
    nx.spiral_layout,
]

g = nx.convert_matrix.from_numpy_matrix(A)
for layout in all_layouts:
    nx.draw(g, node_color=col, node_size=50, pos=layout(g))
    plt.show()
