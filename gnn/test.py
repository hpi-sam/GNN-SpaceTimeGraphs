import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from gnn.models import GCN, SLGCN, GCRNN
from gnn.utils import load_adjacency_matrix, save_model_to_path
from gnn.dataset import TrafficDataset
from gnn.argparser import parse_arguments
from tqdm import tqdm


MODEL_SAVE_PATH = "./saved_models/"

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
        losses.append(loss.item())
        bar.set_description('loss: {:.4f}'.format(loss.item()))
    return np.mean(losses)

if __name__=='__main__':
    parser = parse_arguments()
    args = parser.parse_args()

    if args.gpu:
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        DEVICE = torch.device("cpu")

    dataset_test = TrafficDataset(args, split='test')

    # use data loader
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=1)

    # load adjacency matrix
    adj = load_adjacency_matrix(args, DEVICE)

    if args.model == 'SLGCN':
        model = SLGCN(adj, args, device=DEVICE)
    elif args.model == 'RGCNN':
        model = GCRNN(adj, args, device=DEVICE)
    else:
        model = GCN(adj, args, device=DEVICE)

    model.load_state_dict(torch.load(MODEL_SAVE_PATH + 'slgcn_global.pt'))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print('Iterate over the test-split...')
    ml_test = run_epoch(model=model, optimizer=optimizer, dataloader=dataloader_test, training=False)
    print('Mean loss over test dataset: {:.4f}'.format(ml_test))
