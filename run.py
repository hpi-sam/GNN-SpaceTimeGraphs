import numpy as np
import logging
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from gnn.argparser import parse_arguments
from gnn.dataset import TrafficDataset
from gnn.models import P3D
from gnn.backlog.models import GCN, STGCN, SLGCN, GCRNN
from utils import load_adjacency_matrix, save_model_to_path, get_device

logger = logging.getLogger(__name__)
MODEL_SAVE_PATH = "./saved_models/"

parser = parse_arguments()
args = parser.parse_args()
DEVICE = get_device(args.gpu)


def run_epoch(model, optimizer, dataloader, training=True):
    mu, std = dataloader.dataset.mu, dataloader.dataset.std
    mu = torch.tensor(mu, device=DEVICE)
    std = torch.tensor(std, device=DEVICE)
    bar = tqdm(dataloader)
    losses = []
    if training:
        model.train()
    else:
        model.eval()
    # print("epoch: {}".format(epoch + 1))
    for sample_batched in bar:
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
    print(args)

    # load adjacency matrix
    adj = load_adjacency_matrix(args, DEVICE)

    # Model and optimizer
    model = globals()[args.model](adj, args).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.mode == 'train':
        dataset_train = TrafficDataset(args, split='train')
        dataset_val = TrafficDataset(args, split='val')
        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=1)
        dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=1)

        # Training
        hist_loss = []
        if args.log_file:
            logging.basicConfig(filename=args.log_file, level=logging.INFO)
        else:
            logging.basicConfig(level=logging.INFO, format='# %(message)s')

        for epoch in range(args.n_epochs):
            ml_train = run_epoch(model, optimizer, dataloader_train)
            logger.info(f"epoch: {epoch}")
            logger.info('Mean train-loss over batch: {:.4f}'.format(ml_train))

            ml_val = run_epoch(model, optimizer, dataloader_val, training=False)
            logger.info('Mean validation-loss over batch: {:.4f}'.format(ml_val))
            hist_loss.append((ml_train, ml_val))

        # save the model
        save_model_to_path(args, model)

        np.save(f"losses_on_{args.n_epochs}_epochs", hist_loss)

    if args.mode == 'test':
        dataset_test = TrafficDataset(args, split='test')
        dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=1)

        model.load_state_dict(torch.load(MODEL_SAVE_PATH + 'slgcn_global.pt'))
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        logger.info('Iterate over the test-split...')
        ml_test = run_epoch(model, optimizer, dataloader_test, training=False)
        logger.info('Mean loss over test dataset: {:.4f}'.format(ml_test))
