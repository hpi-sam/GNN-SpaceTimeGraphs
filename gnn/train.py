from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import pathlib as path
import pickle
from tqdm import tqdm

from torch.utils.data import DataLoader
from gnn.models import GCN
from gnn.utils import normalize
from gnn.dataset import TrafficDataset


def train(epochs, model, optimizer, dataloader):
    for epoch in range(epochs):
        bar = tqdm(dataloader)
        losses = []
        for sample_batched in bar:
            model.train()
            optimizer.zero_grad()
            x = torch.tensor(sample_batched['features'], device=device, dtype=torch.float32)
            y = torch.tensor(sample_batched['labels'], device=device, dtype=torch.float32)
            output = model(x, adj)
            loss_train = F.mse_loss(output, y)
            loss_train.backward()
            optimizer.step()
            losses.append(loss_train.item())

            bar.set_description('epoch: {}, loss_train: {:.4f}'.format(epoch + 1, loss_train.item()))
        print('mean loss over batch: {:.4f}'.format(np.mean(losses)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_file", type=str, default="./data/metr_la/train.npz", help="File containing the training data"
    )
    parser.add_argument(
        "--test_file", type=str, default="./data/metr_la/test.npz", help="File containing the testing data"
    )
    parser.add_argument(
        '--batch_size', type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        '--n_epochs', type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        '--pickled_files', type=str, default="metr_la/adj_mx_la.pkl", help="File containing the adjacency matrix"
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    place = args.pickled_files
    place_path = path.Path("./data") / place
    with open(place_path, "rb") as f:
        _, _, adj = pickle.load(f, encoding='latin-1')
    adj = torch.tensor(normalize(adj), device=device)

    # Dataset
    dataset = TrafficDataset(args)

    # use data loader
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=1)

    # Model and optimizer
    model = GCN(nfeat=dataset.features_train.shape[2],
                nhid=100,
                nclass=1,
                n=207).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=.01, weight_decay=0.95)

    model.train()
    train(args.n_epochs, model, optimizer, dataloader)