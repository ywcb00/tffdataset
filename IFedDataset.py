from tffdataset.DatasetUtils import DatasetID

from abc import ABC, abstractmethod
from enum import Enum

class PartitioningScheme(Enum):
    RANGE       = 1
    RANDOM      = 2
    ROUND_ROBIN = 3
    DIRICHLET   = 4

class IFedDataset(ABC):
    def __init__(self, config):
        self.config = config
        self.train = None
        self.val = None
        self.test = None

    @classmethod
    def getFedDataset(self_class, config):
        match config["dataset_id"]:
            case DatasetID.FloodNet | DatasetID.Mnist | DatasetID.Iris:
                from tffdataset.FedTensorFlowDataset import FedTensorFlowDataset
                return FedTensorFlowDataset(config)
            case DatasetID.IrisPyTorch:
                from tffdataset.FedPyTorchDataset import FedPyTorchDataset
                return FedPyTorchDataset(config)
            case _:
                raise NotImplementedError

    def construct(self, dataset, seed=None):
        self.logger.info(f'Partitioning the dataset to {self.config["num_workers"]} ' +
            f'partitions with {self.config["partitioning_scheme"].name} scheme')
        # partition the data
        self.train = self.partitionData(dataset.train, self.config, dataset.num_classes,
            seed=seed, logger=self.logger)
        self.val = self.partitionData(dataset.val, self.config, dataset.num_classes,
            seed=seed, logger=self.logger)
        self.test = self.partitionData(dataset.test, self.config, dataset.num_classes,
            seed=seed, logger=self.logger)
        self.batch_size = dataset.batch_size

    @abstractmethod
    def batch(self):
        pass

    # ===== partitioning =====
    @classmethod
    def partitionData(self_class, data, config, n_classes, seed, logger):
        if(not seed):
            seed = config["seed"]
        n_workers = config["num_workers"]
        match config["partitioning_scheme"]:
            case PartitioningScheme.RANGE:
                data_parts = self_class.partitionDataRange(data, n_workers)
                return data_parts
            case PartitioningScheme.RANDOM:
                data_parts = self_class.partitionDataRandom(data, n_workers, seed)
                return data_parts
            case PartitioningScheme.ROUND_ROBIN:
                data_parts = self_class.partitionDataRoundRobin(data, n_workers)
                return data_parts
            case PartitioningScheme.DIRICHLET:
                data_parts = self_class.partitionDataDirichlet(data, n_workers,
                    n_classes, config["partitioning_alpha"], seed, logger)
                return data_parts

    @classmethod
    @abstractmethod
    def partitionDataRange(self_class, data, n_workers):
        pass

    @classmethod
    @abstractmethod
    def partitionDataRandom(self_class, data, n_workers, seed):
        pass

    @classmethod
    @abstractmethod
    def partitionDataRoundRobin(self_class, data, n_workers):
        pass

    # similar to https://github.com/alpemreacar/FedDyn/blob/48a19fac440ef079ce563da8e0c2896f8256fef9/utils_dataset.py#L140
    @classmethod
    @abstractmethod
    def partitionDataDirichlet(self_class, data, n_partitions, n_classes, alpha,
        seed, logger):
        pass
