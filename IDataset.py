from abc import ABC, abstractmethod

class IDataset(ABC):
    def __init__(self, config, batch_size):
        self.config = config
        self.train = None
        self.val = None
        self.test = None
        self.batch_size = batch_size # set this member when calling the parent constructor

    @abstractmethod
    def load(self):
        pass

    def batch(self):
        self.train = self.train.batch(self.batch_size)
        self.val = self.val.batch(self.batch_size)
        self.test = self.test.batch(self.batch_size)
