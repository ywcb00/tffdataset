from tffdataset.DirectPyTorchDataset import DirectPyTorchDataset
from tffdataset.IFedDataset import IFedDataset

import logging
import numpy as np
import pandas as pd
import queue
import torch

class FedPyTorchDataset(IFedDataset):
    def __init__(self, config):
        super().__init__(config)
        self.logger = logging.getLogger("dataset/FedPyTorchDataset")
        self.logger.setLevel(config["log_level"])

    def batch(self):
        # batch the data partitions
        # FIXME: drop_last was set to true because otherwise the serialized and deserialized model throws an assertion error
        self.train = [torch.utils.data.DataLoader(fp, batch_size=self.batch_size, drop_last=True) for fp in self.train]
        self.val = [torch.utils.data.DataLoader(fp, batch_size=self.batch_size, drop_last=True) for fp in self.val]
        self.test = [torch.utils.data.DataLoader(fp, batch_size=self.batch_size, drop_last=True) for fp in self.test]

    # ===== partitioning =====
    @classmethod
    def partitionDataRange(self_class, data, n_workers):
        n_rows = len(data)
        distribute_remainder = lambda idx: 1 if idx < (n_rows % n_workers) else 0
        data_parts = list()
        current_position = 0
        for w_idx in range(n_workers):
            num_elements = (n_rows // n_workers) + distribute_remainder(w_idx)
            data_parts.append(data[current_position:current_position+num_elements])
            current_position += num_elements
        return data_parts

    @classmethod
    def partitionDataRandom(self_class, data, n_workers, seed):
        # FIXME: reproducibility
        data_parts = torch.utils.data.random_split(data, np.repeat(1.0/n_workers, n_workers))
        return data_parts

    @classmethod
    def partitionDataRoundRobin(self_class, data, n_workers):
        data_parts = list()
        for w_idx in range(n_workers):
            part_tuple = data[list(filter(lambda idx: idx % n_workers == w_idx, range(len(data))))]
            data_parts.append(torch.utils.data.TensorDataset(part_tuple[0], part_tuple[1]))
        return data_parts

    # similar to https://github.com/alpemreacar/FedDyn/blob/48a19fac440ef079ce563da8e0c2896f8256fef9/utils_dataset.py#L140
    @classmethod
    def partitionDataDirichlet(self_class, data, n_partitions, n_classes, alpha,
        seed, logger):
        np.random.seed(seed)
        responses = data.tensors[1]
        class_data_indices = dict()
        class_lookup_dict = dict()
        newclass_index = 0
        for idx, response_tensor in enumerate(responses):
            response = response_tensor.numpy()
            if(not class_lookup_dict or
                not np.any(np.all(response == np.array(list(class_lookup_dict.values())), axis=1))):
                class_lookup_dict[newclass_index] = response
                class_data_indices[newclass_index] = queue.Queue()
                newclass_index += 1
            class_data_indices[next(filter(lambda key: np.all(class_lookup_dict[key] == response),
                class_lookup_dict))].put(idx)

        n_rows = len(data)
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

        partitions = [data[pai] for pai in partition_assigned_indices.values()]
        partitions = [torch.utils.data.TensorDataset(part[0], part[1]) for part in partitions]

        logger.debug(f'Partitioned data into {n_partitions} partitions with ' +
            f'{list(partition_num_elements.values())} respectively with the following ' +
            "distribution of target labels:\n" +
            pd.DataFrame.from_dict(partition_assigned_classes).sort_index().to_string())

        return partitions

    def getPartition(self, partition_index):
        partition_dataset = DirectPyTorchDataset(self.batch_size, None,
            self.train[partition_index],
            self.val[partition_index],
            self.test[partition_index],
            self.config)
        return partition_dataset
