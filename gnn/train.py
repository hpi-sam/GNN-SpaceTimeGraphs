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
import tqdm

from torch.utils.data import DataLoader
from gnn.models import GCN
from gnn.utils import normalize
from gnn.dataset import TrafficDataset

parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_file", type=str, default="./data/metr_la/train.npz", help="File containing the training data"
)
parser.add_argument(
    "--test_file", type=str, default="./data/metr_la/test.npz", help="File containing the testing data"
)
parser.add_argument(
    '--batch_size', type=int, default=1, help="Batch size for training"
)
parser.add_argument(
    '--n_epochs', type=int, default=1, help="Number of training epochs"
)
parser.add_argument(
    '--pickled_files', type=str, default="metr_la/adj_mx_la.pkl", help="File containing the adjacency matrix"
)
args = parser.parse_args()

place = args.pickled_files
place_path = path.Path("./data") / place
with open(place_path, "rb") as f:
    _, _, adj = pickle.load(f, encoding='latin-1')
adj = torch.Tensor(normalize(adj))



# Dataset
dataset = TrafficDataset(args)

# use data loader
dataloader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=8)

# Model and optimizer
model = GCN(nfeat=dataset.features_train.shape[2],
            nhid=100,
            nclass=1,
            n=1)
optimizer = optim.Adam(model.parameters(),
                       lr=0.1, weight_decay=0.95)


def train(epoch):
    t = time.time()

    for i_batch, sample_batched in enumerate(dataset):
        model.train()
        optimizer.zero_grad()
        x = torch.Tensor(sample_batched['features'])
        output = model(x, adj)
        loss_train = F.mse_loss(output, x)
        loss_train.backward()
        optimizer.step()

        print('batch: {:04d}'.format(i_batch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'time: {:.4f}s'.format(time.time() - t))


for epoch in range(args.n_epochs):
    train(epoch)
