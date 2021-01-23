import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from gnn.argparser import parse_arguments
from gnn.dataset import TrafficDataset
from gnn.models import GCN, GCRNN, SLGCN
from gnn.utils import load_adjacency_matrix, save_model_to_path


def run_epoch(model, optimizer, dataloader, training=True):
    bar = tqdm(dataloader)
    losses = []
    for sample_batched in bar:
        model.train()
        optimizer.zero_grad()
        x = torch.tensor(sample_batched['features'],
                         device=DEVICE,
                         dtype=torch.float32)
        y = torch.tensor(sample_batched['labels'],
                         device=DEVICE,
                         dtype=torch.float32)
        output = model(x)
        loss = F.mse_loss(output, y)
        if training:
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        bar.set_description('epoch: {}, loss: {:.4f}'.format(
            epoch + 1, loss.item()))
    return np.mean(losses)


if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()

    if args.gpu:
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        DEVICE = torch.device("cpu")

    print(DEVICE)

    # Dataset
    dataset_train = TrafficDataset(args, split='train')
    dataset_val = TrafficDataset(args, split='val')

    # use data loader
    dataloader_train = DataLoader(dataset_train,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=1)
    dataloader_val = DataLoader(dataset_val,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=1)

    # load adjacency matrix
    adj = load_adjacency_matrix(args, DEVICE)

    # Model and optimizer
    # TODO: model.to(device) instead of passing device as argument
    if args.model == 'SLGCN':
        model = SLGCN(adj, args, device=DEVICE)
    elif args.model == 'RGCNN':
        model = GCRNN(adj, args, device=DEVICE)
    else:
        model = GCN(adj, args, device=DEVICE)

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

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
