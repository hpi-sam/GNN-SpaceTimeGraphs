from __future__ import division
from __future__ import print_function
import sys
import time
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import pathlib as path
import pickle
#from tqdm import tqdm

from torch.utils.data import DataLoader
from gnn.models import MultiTempGCN
from gnn.utils import normalize
from gnn.dataset import TrafficDataset
from gnn.argparser import parse_arguments

# DEFAULT VALUES
SAVE_PATH = './saved_models/'
DEVICE = torch.device("cpu")

def train(epochs, model, optimizer, dataloader):
    hist_loss = []
    for epoch in range(epochs):
        bar = tqdm(dataloader)
        losses = []
        for sample_batched in bar:
            model.train()
            optimizer.zero_grad()
            x = torch.tensor(sample_batched['features'], device=DEVICE, dtype=torch.float32)
            y = torch.tensor(sample_batched['labels'], device=DEVICE, dtype=torch.float32)
            output = model(x, adj)
            loss_train = F.mse_loss(output, y)
            loss_train.backward()
            optimizer.step()
            losses.append(loss_train.item())

            bar.set_description('epoch: {}, loss_train: {:.4f}'.format(epoch + 1, loss_train.item()))
        mean_loss = np.mean(losses)
        hist_loss.append(mean_loss)
        print('mean loss over batch: {:.4f}'.format(mean_loss))
    return hist_loss


if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    if args.gpu:
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        DEVICE = torch.device("cpu")

    place = args.pickled_files
    place_path = path.Path("./data") / place
    with open(place_path, "rb") as f:
        _, _, adj = pickle.load(f, encoding='latin-1')
    adj = torch.tensor(normalize(adj), device=DEVICE)

    # Dataset
    dataset = TrafficDataset(args)  # TODO: create apprrpriate datasets

    # use data loader
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=1)

    # Model and optimizer
    model = MultiTempGCN(nfeat=dataset.features_train.shape[2],
                         nhid=100,
                         nclass=1,
                         n=207, device=DEVICE)

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    MODEL_SAVE_PATH = "./saved_models/"
    if args.model_name is not None:
        filepath = MODEL_SAVE_PATH + args.model_name + '.pt'
    else:
        filepath = MODEL_SAVE_PATH + 'model_001' + '.pt'
    if path.Path(filepath).is_file():
        filepath = filepath.replace(filepath[-6:-3], '{0:03}'.format(int(filepath[-6:-3])+1))

    torch.save(model.state_dict(), filepath)
    hist_loss = train(args.n_epochs, model, optimizer, dataloader)
    np.save(f"losses_on_{args.n_epochs}_epochs", hist_loss)
