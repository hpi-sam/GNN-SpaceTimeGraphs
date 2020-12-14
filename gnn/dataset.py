from gnn.utils import load_data
from torch.utils.data import Dataset
import torch


class TrafficDataset(Dataset):
    def __init__(self, args):
        self.features_train, self.labels_train = load_data(args.train_file)

    def __len__(self):
        return len(self.features_train)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return {"features": self.features_train[idx, :, :], "train": self.labels_train[idx, :, :]}

