from gnn.argparser import parse_arguments
from utils import load_adjacency_matrix, get_device
from gnn.dataset import TrafficDataset
from torch.utils.data import DataLoader
from gnn import models
import torch.optim as optim
from run import run_epoch
import optuna
import logging
import inspect
import re
from datetime import datetime

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

    @staticmethod
    def get_list_size(lst):
        if len(lst) > 2:
            return "categorical"
        elif len(lst) == 2:
            return "range"
        else:
            raise ValueError("list should be either a range (2 elements) or categorical (3+ elements)")

    def get_tunable_parameters(self, trial, args):
        type_to_suggestion_map = {int: trial.suggest_int, float: trial.suggest_float}
        tune_param = {}
        for key, val in inspect.getmembers(args):
            if self.ht_var.match(key) and val:
                sugest_type = self.get_list_size(val)
                if sugest_type == "categorical":
                    tune_param[self.ht_var.sub("", key)] = trial.suggest_categorical(key, val)
                if sugest_type == "range":
                    tune_param[self.ht_var.sub("", key)] = type_to_suggestion_map[self.get_list_type(val)](key, *val)
        tune_param["spatial_channels"] = int(tune_param["bottleneck_channels"] * tune_param["spatial_channels"])
        return tune_param

    def objective(self, trial):
        for (param, value) in self.get_tunable_parameters(trial, self.args).items():
            setattr(self.args, param, value)

        model = getattr(models, args.model)(self.adj, self.args).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr)
        # Training
        val_loss = 0
        if args.log_file:
            logging.basicConfig(filename=args.log_file, level=logging.INFO)
        else:
            logging.basicConfig(level=logging.INFO, format='# %(message)s')
        val_loss_list = []
        for epoch in range(self.args.n_epochs):
            logger.info(f"epoch: {epoch}")
            train_loss = run_epoch(model, optimizer, self.dataloader_train)
            logger.info('Mean train-loss over batch: {:.4f}'.format(train_loss))
            val_loss = run_epoch(model, optimizer, self.dataloader_val, training=False)
            logger.info('Mean validation-loss over batch: {:.4f}'.format(val_loss))
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
            val_loss_list.append(val_loss)
        return min(val_loss_list)


if __name__ == '__main__':
    parser = parse_arguments()
    args = parser.parse_args()

    objective = ObjectiveCreator(args).objective

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(n_startup_trials=3),
                                pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource='auto',
                                                                              reduction_factor=4,
                                                                              min_early_stopping_rate=0))
    study.optimize(objective, n_trials=args.n_trials)
    df_study = study.trials_dataframe()
    tstamp = datetime.now().strftime("%Y-%m-%dT%H:%M")
    mode = 'free' if args.learnable_l else 'base'
    df_study.to_csv(f'./studies/{args.model}-{args.convolution_operator}-{tstamp}-{mode}.csv')
