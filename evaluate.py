import torch
import logging
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from gnn.argparser import parse_arguments
from gnn.dataset import TrafficDataset
from gnn.models import P3D
from utils import load_adjacency_matrix, get_device


logger = logging.getLogger(__name__)
MODEL_SAVE_PATH = "./saved_models/pems_bay/"

parser = parse_arguments()
args = parser.parse_args()
DEVICE = get_device(args.gpu)


def evaluate(m, dataloader):
    mu, std = dataloader.dataset.mu, dataloader.dataset.std
    mu = torch.tensor(mu, device=DEVICE)
    std = torch.tensor(std, device=DEVICE)
    bar = tqdm(dataloader)

    outputs = np.array([])
    targets = np.array([])

    # put model in evaluation mode to remove dropout
    model.eval()
    for batch in bar:
        x = batch['features'].to(DEVICE).type(torch.float32)
        y = batch['labels'].to(DEVICE).type(torch.float32)
        output = m(x)
        output_denormalized = output * std + mu
        y_denormalized = y * std + mu

        # logg outputs and targets
        outputs = np.append(outputs, output_denormalized.detach().cpu().numpy())
        targets = np.append(targets, y_denormalized.detach().cpu().numpy())

    return outputs, targets


def logg_stats(predictions, targets):
    error = predictions - targets
    rmse = np.sqrt(np.mean(np.square(error)))
    mae = np.mean(np.abs(error))
    mape = np.mean(np.abs(error/targets)[targets > 1e-3]) * 100

    logger.info('RMSE: {:.4f}'.format(rmse))
    logger.info('MAE: {:.4f}'.format(mae))
    logger.info('MAPE: {:.4f}'.format(mape))


if __name__ == '__main__':
    if args.log_file:
        logging.basicConfig(filename=args.log_file, level=logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO, format='# %(message)s')

    # load adjacency matrix
    adj = load_adjacency_matrix(args, DEVICE)

    # Model and optimizer
    model = P3D(adj, args).to(DEVICE)

    # training data
    dataset_train = TrafficDataset(args, split='train')
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False, num_workers=1)

    # testing data
    dataset_test = TrafficDataset(args, split='test')
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=1)

    model.load_state_dict(torch.load(MODEL_SAVE_PATH + args.model_name + '.pt'))

    print(f"Evaluating model {args.model_name} on training data...")
    outputs_train, targets_train = evaluate(model, dataloader_train)
    logger.info('Statistics on training data:')
    logg_stats(outputs_train, targets_train)
    print(f"Evaluating model {args.model_name} on testing data...")
    logger.info('Statistics on testing data:')
    outputs_test, targets_test = evaluate(model, dataloader_test)
    logg_stats(outputs_test, targets_test)

    np.save(f"./studies/output/pems_bay/{args.model}-train-output", outputs_train)
    np.save(f"./studies/output/pems_bay/{args.model}-train-target", targets_train)

    np.save(f"./studies/output/pems_bay/{args.model}-test-output", outputs_test)
    np.save(f"./studies/output/pems_bay/{args.model}-test-target", targets_test)
