import torch.utils.data as data
import numpy as np

class Dataset(data.Dataset):
    def __init__(self, examples):
        super(Dataset,self).__init__()
        self.examples = examples 

    def __getitem__(self, idx):
        return self.examples[idx]
        
    def __len__(self):
        return len(self.examples)
