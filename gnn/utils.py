from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import os
import pathlib as path
import pickle

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from datetime import datetime


def generate_graph_seq2seq_io_data(df,
                                   x_offsets,
                                   y_offsets,
                                   add_time_in_day=True,
                                   add_day_in_week=False,
                                   scaler=None):
    """Generate samples from.

    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    data, mean, std = normalize_data(data)

    data_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values -
                    df.index.values.astype("datetime64[D]")) / np.timedelta64(
            1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)

    data = np.concatenate(data_list, axis=-1)
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x, y = [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ..., :1]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)

    return x, y, mean, std


def generate_train_val_test_inst_to_inst(args):
    df = pd.read_hdf(args.traffic_df_filename)
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    data, mean, std = normalize_data(data)
    data_list = [data]
    if args.add_time_in_day:
        time_ind = (df.index.values -
                    df.index.values.astype("datetime64[D]")) / np.timedelta64(
            1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)

    data = np.concatenate(data_list, axis=-1)
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x, y = [], []
    # t is the index of the last observation.
    min_t = 0
    max_t = num_samples - 1
    for t in range(min_t, max_t):
        x_t = data[t, ...]
        y_t = data[t + 1, ..., :1]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    # train
    x_train_iti, y_train_iti = x[:num_train], y[:num_train]
    # val
    x_val_iti, y_val_iti = (
        x[num_train:num_train + num_val],
        y[num_train:num_train + num_val],
    )
    # test
    x_test_iti, y_test_iti = x[-num_test:], y[-num_test:]

    for cat in ["train_iti", "val_iti", "test_iti"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
            mu=mean,
            std=std,
            x_offsets=0,
            y_offsets=1,
        )


def normalize_data(data):
    # apply Z-Score normalization
    mean = np.mean(data, axis=(0, 1))
    std = np.std(data, axis=(0, 1))

    out = data - mean
    out = out / std
    return out, mean, std


def generate_train_val_test(args):
    df = pd.read_hdf(args.traffic_df_filename)
    # 0 is the latest observed sample.
    x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
        np.concatenate((np.arange(-11, 1, 1),)))
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 13, 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y, mean, std = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=False,
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    # train
    x_train_sts, y_train_sts = x[:num_train], y[:num_train]
    # val
    x_val_sts, y_val_sts = (
        x[num_train:num_train + num_val],
        y[num_train:num_train + num_val],
    )
    # test
    x_test_sts, y_test_sts = x[-num_test:], y[-num_test:]

    for cat in ["train_sts", "val_sts", "test_sts"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
            mu=mean,
            std=std,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def compute_normalized_laplacian(adj):
    """ Compute L = D^{-1/2}(D-A)D^{-1/2}, where D denotes the degree matrix, and A is the adjacency matrix
    and L is the normalized laplacian
    """
    adj_sparse = sp.coo_matrix(adj.cpu().numpy())
    d = np.array(adj_sparse.sum(1))
    d_inv_sqrt = np.power(d, -1 / 2).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
    d_inv_sqrt = sp.diags(d_inv_sqrt)
    lhs = sp.eye(adj_sparse.shape[0])
    rhs = adj_sparse.dot(d_inv_sqrt).transpose().dot(d_inv_sqrt).tocoo()
    normalized_laplacian = (lhs - rhs).tocoo()
    normalized_laplacian = torch.sparse.LongTensor(
        torch.LongTensor([normalized_laplacian.row.tolist(), normalized_laplacian.col.tolist()]),
        torch.LongTensor(normalized_laplacian.data.astype(np.int32))
    )
    return normalized_laplacian


def generate_knn_ids(dist, k):
    return torch.argsort(dist, axis=-1)[:, -k - 1:-1]


def load_data(filename):
    npz = np.load(filename)
    features, labels, mu, std = npz['x'], npz['y'], npz['mu'], npz['std']
    return features, labels, mu, std


def load_adjacency_matrix(args, DEVICE):
    place = args.pickled_files
    place_path = path.Path("./data") / place
    with open(place_path, "rb") as f:
        _, _, adj = pickle.load(f, encoding='latin-1')
    adj = torch.tensor(normalize(adj), device=DEVICE)
    return adj


def save_model_to_path(args, model, model_save_path="./saved_models/"):
    if args.model_name is not None:
        filepath = model_save_path + args.model_name + '.pt'
    else:
        filepath = model_save_path + 'model_001' + '.pt'
    if path.Path(filepath).is_file():
        filepath = filepath.replace(filepath[-6:-3],
                                    '{0:03}'.format(int(filepath[-6:-3]) + 1))

    torch.save(model.state_dict(), filepath)


def cycle_encode(time_stamp):
    time_stamp = datetime.fromtimestamp(time_stamp)
    #   time_feats = [(time_stamp.month, 12),
    #                 (time_stamp.day, 31),
    #                 (time_stamp.weekday(), 6),
    #                 (time_stamp.hour, 23),
    #                 (time_stamp.minute, 59),
    #                 (time_stamp.second, 59)]
    time_feats = [(time_stamp.month, 12), (time_stamp.day, 31),
                  (time_stamp.weekday(), 6), (time_stamp.hour, 23),
                  (time_stamp.minute, 59), (time_stamp.second, 59)]
    encoded = list()
    for time_feat, period in time_feats:
        # sin_feat = np.sin(time_feat/period)
        # cos_feat = np.cos(time_feat/period)
        sin_feat = np.sin(time_feat / period)
        cos_feat = np.cos(time_feat / period)
        encoded = encoded + [sin_feat, cos_feat]
    return encoded


def main(args):
    print("Generating training data")
    if args.sts:
        generate_train_val_test(args)
    else:
        generate_train_val_test_inst_to_inst(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",
                        type=str,
                        default="data/",
                        help="Output directory.")
    parser.add_argument("--add_time_in_day",
                        type=int,
                        default=1,
                        help="Output directory.")
    parser.add_argument(
        "--sts",
        type=bool,
        default=False,
        help="True to generate Seq_to_seq data and false to create Inst_to_inst"
    )
    parser.add_argument(
        "--traffic_df_filename",
        type=str,
        default="data/metr-la.h5",
        help="Raw traffic readings.",
    )
    args = parser.parse_args()
    main(args)
