from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from gnn.models import GCN
from gnn.utils import load_data

parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_file", type=str, default="data/metr_la_train.npz", help="File containing the training data"
)
parser.add_argument(
    "--test_file", type=str, default="data/metr_la_test.npz", help="File containing the testing data"
)
args = parser.parse_args()

# Load data
adj, features_train, labels_train = load_data(args.train_file)
_, features_test, labels_test = load_data(args.test_file)

# Dataset

# use data loader

# Model and optimizer
model = GCN(nfeat=features_train.shape[2],
            nhid=args.hidden,
            nclass=1,
            n=features_test.shape[1])
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

# loss function (l2-loss)

# backprop
