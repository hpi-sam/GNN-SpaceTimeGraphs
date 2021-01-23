from gnn.utils import load_data
from torch.utils.data import Dataset
import torch


class TrafficDataset(Dataset):
    def __init__(self, args, split='train'):
        if split == 'train':
            source_file = args.train_file
        elif split == 'val':
            source_file = args.val_file
        else:
            source_file = args.test_file
        self.features_train, self.labels_train = load_data(source_file)
        # forecast_horizon: number of time-steps of 5 Minute to intervals to predict in the future; 3 ~ 15 Min
        if args.forecast_horizon >= 1 and len(self.labels_train.shape) == 4:
            self.labels_train = self.labels_train[:, args.forecast_horizon - 1, :, :]
        # create the toy data for only 5 nodes
        if args.toy_data: 
            self.features_train = self.features_train[:int(0.025*self.features_train.shape[0]), :, :]
            self.labels_train = self.labels_train[:int(0.025*self.labels_train.shape[0]), :, :]

    def __len__(self):
        return len(self.features_train)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return {"features": self.features_train[idx, :, :], "labels": self.labels_train[idx, :, :]}

