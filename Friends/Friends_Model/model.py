import numpy as np
import torch as T
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()

    def forward(self, batch):
        x,y = batch
        out = None
        return out

    def get_state(self):
        pass