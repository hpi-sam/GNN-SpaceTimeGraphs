import pandas as pd
import numpy as np
import pickle
import argparse


def generate_sensor_positions(sensor_loc_filename, adj_mx):
    df = pd.read_csv(sensor_loc_filename)
    with open(adj_mx, 'rb') as f:
        # maps a sensor_id to an index in the distance data-frame df
        sensor_id_to_ind = pickle.load(f, encoding='utf-8')[1]
    keys = list(sensor_id_to_ind.keys())
    postions = np.array([[df[df['sensor_id'] == int(key)]['longitude'].values[0],
                          df[df['sensor_id'] == int(key)]['latitude'].values[0]] for key in keys])
    return postions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensor_loc_filename', type=str, default='../data/pems_bay/graph_sensor_locations_bay.csv',
                        help='Comma separated file containing the location longitudes and latitudes for sensor ids')
    parser.add_argument('--adj_mx', type=str, default='../data/pems_bay/adj_mx_bay.pkl',
                        help='Pickle file containing the adjacency matrix')
    parser.add_argument('--output_npy_filename', type=str, default='../data/pems_bay/pos_bay.npy',
                        help='Target filename: *.npy')
    args = parser.parse_args()

    # generate positional data np.array([long_i, lat_i]) i in [0, #graph_ids - 1]
    pos = generate_sensor_positions(args.sensor_loc_filename, args.adj_mx)
    # save to .npy file
    np.save(args.output_npy_filename, pos)
