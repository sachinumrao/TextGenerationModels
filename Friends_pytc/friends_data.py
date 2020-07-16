import torch 
import numpy as np
import config 


class FriendsDataset(torch.utils.data.Dataset):
    def __init__(self, data_file):
        self.data = np.load(data_file, allow_pickle=False)
        self.max_len = config.max_len

    def __len__(self):
        n = self.data.shape[0]
        return n

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx, :-1])
        y = torch.tensor(self.data[idx, 1:])
        return (x,y)