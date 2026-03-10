from tffdataset.IDataset import IDataset

from abc import abstractmethod

class ITensorFlowDataset(IDataset):
    def __init__(self, config, batch_size):
        super().__init__(config, batch_size)

    def batch(self):
        self.train = self.train.batch(self.batch_size)
        self.val = self.val.batch(self.batch_size)
        self.test = self.test.batch(self.batch_size)
