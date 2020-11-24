import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('data/graph_sensor_locations_la.csv')
xs = df['longitude']
ys = df['latitude']
with open('data/adj_mx.pkl', 'rb') as f:
    [sensor_ids, sensor_id_to_ind, A] = pickle.load(f, encoding='utf-8')
keys = list(sensor_id_to_ind.keys())
pos = np.array([[df[df['sensor_id'] == int(key)]['longitude'].values[0], df[df['sensor_id'] == int(key)]['latitude'].values[0]] for key in keys])
np.save('data/pos_la.npy', pos)