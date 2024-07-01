from enum import Enum
import logging

class PartitioningScheme(Enum):
    RANGE       = 1
    RANDOM      = 2
    ROUND_ROBIN = 3

class FedDataset():
    def __init__(self, config):
        self.config = config
        self.train = None
        self.val = None
        self.test = None
        self.logger = logging.getLogger("dataset/FedDataset")
        self.logger.setLevel(config["log_level"])

    def construct(self, dataset):
        self.logger.info(f'Partitioning the dataset to {self.config["num_workers"]} ' +
            f'partitions with {self.config["part_scheme"].name} scheme')
        # partition the data
        self.train = self.partitionData(dataset.train, self.config)
        self.val = self.partitionData(dataset.val, self.config)
        self.test = self.partitionData(dataset.test, self.config)
        self.batch_size = dataset.batch_size

    def batch(self):
        # batch the data partitions
        self.train = [fp.batch(self.batch_size) for fp in self.train]
        self.val = [fp.batch(self.batch_size) for fp in self.val]
        self.test = [fp.batch(self.batch_size) for fp in self.test]

    # ===== partitioning =====
    @classmethod
    def partitionData(self_class, data, config):
        n_workers = config["num_workers"]
        match config["part_scheme"]:
            case PartitioningScheme.RANGE:
                data_parts = partitionDataRange(data, n_workers)
                return data_parts
            case PartitioningScheme.RANDOM:
                data.shuffle(data.cardinality(), seed=config["seed"])
                data_parts = partitionDataRange(data, n_workers)
                return data_parts
            case PartitioningScheme.ROUND_ROBIN:
                data_parts = [data.shard(n_workers, w_idx) for w_idx in range(n_workers)]
                return data_parts

    @classmethod
    def partitionDataRange(self_class, data, n_workers):
        n_rows = data.cardinality().numpy()
        distribute_remainder = lambda idx: 1 if idx < (n_rows % n_workers) else 0
        data_parts = list()
        num_elements = 0
        for w_idx in range(n_workers):
            data.skip(num_elements)
            num_elements = (n_rows // n_workers) + distribute_remainder(w_idx)
            data_parts.append(data.take(num_elements))
        return data_parts
