from gnn.utils import load_data
from torch.utils.data import Dataset
import torch


class TrafficDataset(Dataset):
    def __init__(self, args, time_steps=1):
        self.time_steps = time_steps - 1
        self.features_train, self.labels_train = load_data(args.train_file)
        if time_steps > 0:
            self.features_train = self.features_train[:-time_steps, :, :]

    def __len__(self):
        return len(self.features_train)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label_idx = idx + self.time_steps
        return {"features": self.features_train[idx, :, :], "labels": self.labels_train[label_idx, :, :]}

