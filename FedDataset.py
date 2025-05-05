from enum import Enum
import logging
import numpy as np
import pandas as pd
import queue
import tensorflow as tf

class PartitioningScheme(Enum):
    RANGE       = 1
    RANDOM      = 2
    ROUND_ROBIN = 3
    DIRICHLET   = 4

class FedDataset():
    def __init__(self, config):
        self.config = config
        self.train = None
        self.val = None
        self.test = None
        self.logger = logging.getLogger("dataset/FedDataset")
        self.logger.setLevel(config["log_level"])

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

    def batch(self):
        # batch the data partitions
        self.train = [fp.batch(self.batch_size) for fp in self.train]
        self.val = [fp.batch(self.batch_size) for fp in self.val]
        self.test = [fp.batch(self.batch_size) for fp in self.test]

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
                    n_classes, config["partitioning_dirichlet_alpha"], seed, logger)
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

    @classmethod
    def partitionDataRandom(self_class, data, n_workers, seed):
        data.shuffle(data.cardinality(), seed=seed)
        data_parts = self_class.partitionDataRange(data, n_workers)
        return data_parts

    @classmethod
    def partitionDataRoundRobin(self_class, data, n_workers):
        data_parts = [data.shard(n_workers, w_idx) for w_idx in range(n_workers)]
        return data_parts

    # similar to https://github.com/alpemreacar/FedDyn/blob/48a19fac440ef079ce563da8e0c2896f8256fef9/utils_dataset.py#L140
    @classmethod
    def partitionDataDirichlet(self_class, data, n_partitions, n_classes, alpha,
        seed, logger):
        np.random.seed(seed)
        iter_data = data.as_numpy_iterator()
        class_data_indices = dict()
        class_lookup_dict = dict()
        newclass_index = 0
        for idx, elem in enumerate(iter_data):
            response = elem[1]
            if(not class_lookup_dict or
                not np.any(np.all(response == np.array(list(class_lookup_dict.values())), axis=1))):
                class_lookup_dict[newclass_index] = response
                class_data_indices[newclass_index] = queue.Queue()
                newclass_index += 1
            class_data_indices[next(filter(lambda key: np.all(class_lookup_dict[key] == response),
                class_lookup_dict))].put(idx)

        n_rows = data.cardinality().numpy()
        distribute_remainder = lambda idx: 1 if idx < (n_rows % n_partitions) else 0
        # balanced cardinality
        partition_num_elements = {p_idx: (n_rows // n_partitions) + distribute_remainder(p_idx)
            for p_idx in range(n_partitions)}

        # draw the class priors for all partitions from a dirichlet distribution
        class_priors = np.random.dirichlet(alpha=[alpha]*n_classes, size=n_partitions)
        partition_assigned_classes = dict()
        partition_assigned_indices = dict()
        for _ in range(n_rows):
            # randomly select the partition to which to add an element
            current_partition = np.random.choice([p_idx for p_idx, n_elements
                in partition_num_elements.items() if n_elements > 0])
            current_priors = [class_priors[current_partition][c_idx] * (not class_data_indices[c_idx].empty())
                    for c_idx in class_data_indices.keys()]
            current_priors = np.array(current_priors) / np.sum(current_priors)
            assign_class = np.random.choice(list(class_data_indices.keys()),
                p=current_priors)
            partition_assigned_classes[current_partition][assign_class] = partition_assigned_classes.setdefault(
                current_partition, dict()).setdefault(assign_class, 0) + 1
            partition_assigned_indices.setdefault(current_partition, list()).append(
                class_data_indices[assign_class].get())
        partition_assigned_indices = {p_idx: partition_assigned_indices[p_idx] for p_idx in range(n_partitions)}

        partitions = list()
        for p_idx, pai in partition_assigned_indices.items():
            part_x, part_y = zip(*list(map(list(data).__getitem__, pai)))
            part_x = np.array(part_x)
            part_y = np.array(part_y)
            part_x = tf.data.Dataset.from_tensor_slices(part_x)
            part_y = tf.data.Dataset.from_tensor_slices(part_y)
            partitions.append(tf.data.Dataset.zip((part_x, part_y)))

        logger.debug(f'Partitioned data into {n_partitions} partitions with ' +
            f'{list(partition_num_elements.values())} respectively with the following ' +
            "distribution of target labels:\n" +
            pd.DataFrame.from_dict(partition_assigned_classes).sort_index().to_string())

        return partitions
