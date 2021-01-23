import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import pathlib as path
import pickle
from tqdm import tqdm

from torch.utils.data import DataLoader
from gnn.models import GCN, SLGCN, GCRNN
from gnn.utils import normalize
from gnn.dataset import TrafficDataset
from gnn.argparser import parse_arguments


def run_epoch(model, optimizer, dataloader, training=True):
    bar = tqdm(dataloader)
    losses = []
    for sample_batched in bar:
        model.train()
        optimizer.zero_grad()
        x = torch.tensor(sample_batched['features'], device=DEVICE, dtype=torch.float32)
        y = torch.tensor(sample_batched['labels'], device=DEVICE, dtype=torch.float32)
        output = model(x)
        loss = F.mse_loss(output, y)
        if training:
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        bar.set_description('epoch: {}, loss: {:.4f}'.format(epoch + 1, loss.item()))
    return np.mean(losses)


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
    dataset_train = TrafficDataset(args, split='train')
    dataset_val = TrafficDataset(args, split='val')

    # use data loader
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,
                                  shuffle=True, num_workers=1)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size,
                                shuffle=False, num_workers=1)

    # Model and optimizer
    if args.model == 'SLGCN':
        model = SLGCN(adj,
                      nfeat=dataset_train.features_train.shape[2],
                      nhid=100,
                      nclass=1,
                      N=dataset_train.features_train.shape[1],
                      device=DEVICE)
    elif args.model == 'RGCNN':
        num_nodes = dataset_train.features_train.shape[2]
        input_dim = 2
        num_units = 1000
        hidden_state_size = 264
        nclass = 1
        seqlen = dataset_train.features_train.shape[1]
        model = GCRNN(adj,
                      num_nodes,
                      num_units,
                      input_dim,
                      nclass,
                      hidden_state_size,
                      seqlen).to(DEVICE)
    else:
        model = GCN(adj,
                    nfeat=dataset_train.features_train.shape[2],
                    nhid=100,
                    nclass=1,
                    N=dataset_train.features_train.shape[1],
                    device=DEVICE)

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    # Training
    hist_loss = []
    for epoch in range(args.n_epochs):
        ml_train = run_epoch(model, optimizer, dataloader_train)
        print('Mean train-loss over batch: {:.4f}'.format(ml_train))

        ml_val = run_epoch(model, optimizer, dataloader_val, training=False)
        print('Mean validation-loss over batch: {:.4f}'.format(ml_val))
        hist_loss.append((ml_train, ml_val))

    MODEL_SAVE_PATH = "./saved_models/"
    if args.model_name is not None:
        filepath = MODEL_SAVE_PATH + args.model_name + '.pt'
    else:
        filepath = MODEL_SAVE_PATH + 'model_001' + '.pt'
    if path.Path(filepath).is_file():
        filepath = filepath.replace(filepath[-6:-3], '{0:03}'.format(int(filepath[-6:-3])+1))

    torch.save(model.state_dict(), filepath)
    np.save(f"losses_on_{args.n_epochs}_epochs", hist_loss)
