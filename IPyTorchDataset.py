from tffdataset.IDataset import IDataset

from abc import abstractmethod
import torch

class IPyTorchDataset(IDataset):
    def __init__(self, config, batch_size):
        super().__init__(config, batch_size)

    def batch(self):
        raise NotImplementedError

    @classmethod
    def subsetToDataset(self_class, subset):
        data_tuples = subset.dataset[subset.indices]
        dataset = torch.utils.data.TensorDataset(data_tuples[0], data_tuples[1])
        return dataset

    @classmethod
    @abstractmethod
    def getExampleInputs(self_class):
        # example inputs for the pytorch module (i.e., argument for the
        # `forward(X)' method are required for the serialization of pytorch
        # model through the `export()' function
        pass

    def trainSize(self):
        return len(self.train.dataset)

    def valSize(self):
        return len(self.val.dataset)

    def testSize(self):
        return len(self.test.dataset)
