import torch
import torch.nn as nn

class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def save(self, fname):
        torch.save(self.state_dict(), fname)

    def load(self, fname):
        self.load_state_dict(torch.load(fname))
