import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from gnn.argparser import parse_arguments
from gnn.dataset import TrafficDataset
from gnn.models import P3D
from gnn.backlog.models import GCRNN, SLGCN, STGCN
from gnn.utils import load_adjacency_matrix, save_model_to_path, get_device

parser = parse_arguments()
args = parser.parse_args()
DEVICE = get_device(args.gpu)


def run_epoch(model, optimizer, dataloader, training=True):
    mu, std = dataloader.dataset.mu, dataloader.dataset.std
    mu = torch.tensor(mu, device=DEVICE)
    std = torch.tensor(std, device=DEVICE)
    bar = tqdm(dataloader)
    losses = []
    # print("epoch: {}".format(epoch + 1))
    for sample_batched in bar:
        model.train()
        optimizer.zero_grad()
        x = sample_batched['features'].to(DEVICE).type(torch.float32)
        y = sample_batched['labels'].to(DEVICE).type(torch.float32)
        output = model(x)
        output_denormalized = output * std + mu
        y_denormalized = y * std + mu

        loss = F.mse_loss(output, y)
        loss_mse = F.mse_loss(output_denormalized, y_denormalized)
        loss_mae = F.l1_loss(output_denormalized, y_denormalized)

        if training:
            loss.backward()
            optimizer.step()
        losses.append(loss_mse.item())
        bar.set_description('loss_mae: {:.1f}, loss_mse: {:.1f}'.format(
            loss_mae.item(), loss_mse.item()))
    return np.mean(losses)


if __name__ == "__main__":
    print(DEVICE)
    # Dataset
    dataset_train = TrafficDataset(args, split='train')
    dataset_val = TrafficDataset(args, split='val')

    # use data loader
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=1)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=1)

    # load adjacency matrix
    adj = load_adjacency_matrix(args, DEVICE)

    # Model and optimizer
    model_list = {'SLGCN': SLGCN, 'RGCNN': GCRNN, 'STGCN': STGCN, 'P3D': P3D}
    model = model_list[args.model](adj, args).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training
    hist_loss = []
    for epoch in range(args.n_epochs):
        ml_train = run_epoch(model, optimizer, dataloader_train)
        print('Mean train-loss over batch: {:.4f}'.format(ml_train))

        ml_val = run_epoch(model, optimizer, dataloader_val, training=False)
        print('Mean validation-loss over batch: {:.4f}'.format(ml_val))
        hist_loss.append((ml_train, ml_val))

    # save the model
    save_model_to_path(args, model)

    np.save(f"losses_on_{args.n_epochs}_epochs", hist_loss)
