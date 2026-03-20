from abc import ABC, abstractmethod

class IDataset(ABC):
    def __init__(self, config, batch_size):
        self.config = config
        self.train = None
        self.val = None
        self.test = None
        self.batch_size = batch_size # set this member when calling the parent constructor

    @abstractmethod
    def load(self, seed=None):
        pass

    @abstractmethod
    def batch(self):
        pass

    @abstractmethod
    def trainSize(self):
        pass

    @abstractmethod
    def valSize(self):
        pass
    
    @abstractmethod
    def testSize(self):
        pass
