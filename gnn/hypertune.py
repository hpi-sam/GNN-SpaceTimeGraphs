from gnn.argparser import parse_arguments
from gnn.utils import load_adjacency_matrix, get_device
from gnn.dataset import TrafficDataset
from torch.utils.data import DataLoader
from gnn import models
import torch.optim as optim
from gnn.train import run_epoch
import optuna
import logging
import inspect
import re

logging.basicConfig(level=logging.INFO, format='# %(message)s')
logger = logging.getLogger(__name__)


class ObjectiveCreator:
    def __init__(self, args):
        self.args = args
        self.device = get_device(args.gpu)
        dataset_train = TrafficDataset(args, split='train')
        dataset_val = TrafficDataset(args, split='val')
        self.dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=1)
        self.dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=1)
        self.adj = load_adjacency_matrix(args, self.device)
        self.ht_var = re.compile("^h_")

    @staticmethod
    def get_list_type(lst):
        types = set([type(element) for element in lst])
        if len(types) > 1:
            raise TypeError("List has inconsistent types")
        return types.pop()

    def get_tunable_parameters(self, trial, args):
        type_to_suggestion_map = {int: trial.suggest_int, float: trial.suggest_float}
        tune_param = {self.ht_var.sub("", key): type_to_suggestion_map[self.get_list_type(val)](key, *val)
                      for (key, val) in inspect.getmembers(args) if self.ht_var.match(key)}
        return tune_param

    def objective(self, trial):
        for (param, value) in self.get_tunable_parameters(trial, self.args).items():
            setattr(self.args, param, value)

        model = getattr(models, args.model)(self.adj, self.args, self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr)
        # Training
        val_loss = 0
        for epoch in range(5):

            logger.info(f"epoch: {epoch}")
            logger.info("train")
            train_loss = run_epoch(model, optimizer, self.dataloader_train)
            logger.info('Mean train-loss over batch: {:.4f}'.format(train_loss))
            logger.info("val")
            val_loss = run_epoch(model, optimizer, self.dataloader_val, training=False)
            logger.info('Mean validation-loss over batch: {:.4f}'.format(val_loss))
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return val_loss


if __name__ == '__main__':
    parser = parse_arguments()
    args = parser.parse_args()

    objective = ObjectiveCreator(args).objective

    study = optuna.create_study(direction="minimize",
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=1,
                                                                   n_warmup_steps=2,
                                                                   interval_steps=1))
    study.optimize(objective, n_trials=5)
