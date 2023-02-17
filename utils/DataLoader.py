import random
import numpy as np

class DataLoader():
    def __init__(self, dataset, batch_size=1, shuffle=False):
        assert batch_size > 0
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.lidx = list(range(len(self.dataset)))

        # Padding last batch if necessary
        if len(self.lidx) % self.batch_size != 0:
            self.lidx = self.lidx + random.sample(self.lidx, self.batch_size - (len(self.lidx) % self.batch_size))

        # Shuffle if necessary
        if self.shuffle:
            random.shuffle(self.lidx)

    def __getitem__(self, idx):
        assert idx >= 0
        if idx == 0:
            self.lidx = list(range(len(self.dataset)))

            # Padding last batch if necessary
            if len(self.lidx) % self.batch_size != 0:
                self.lidx = self.lidx + random.sample(self.lidx, self.batch_size - (len(self.lidx) % self.batch_size))

            # Shuffle if necessary
            if self.shuffle:
                random.shuffle(self.lidx)
        if (idx >= len(self.lidx) / self.batch_size):
            return self.dataset[len(self.dataset)]
        idxs = self.lidx[idx*self.batch_size:idx*self.batch_size+self.batch_size]
        return [self.dataset[i] for i in idxs]
