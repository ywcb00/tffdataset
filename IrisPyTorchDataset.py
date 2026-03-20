from tffdataset.IPyTorchDataset import IPyTorchDataset

import csv
import logging
import numpy as np
import os
import pandas as pd
from pathlib import Path
import torch

class IrisPyTorchDataset(IPyTorchDataset):
    num_classes = 3
    element_spec = {"in": {"shape": 4, "dtype": np.float32},
        "out": {"shape": 3, "dtype": np.float32}}
    default_batch_size = 2

    dataset_path = Path(r'./data/iris.csv')
    dataset_url = r'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'

    def __init__(self, config):
        super().__init__(config, batch_size=self.default_batch_size)
        self.logger = logging.getLogger("dataset/IrisPyTorchDataset")
        self.logger.setLevel(config["log_level"])

    def load(self, seed=None):
        if(not seed):
            seed = self.config["seed"]
        if(not self.dataset_path.exists()):
            self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
            df = pd.read_csv(self.dataset_url)
            df.to_csv(self.dataset_path)
        train, val, test = self.getDataset(self.config, seed=seed)
        self.train = train
        self.val = val
        self.test = test
        self.logger.info(f'Found {len(train)} train instances, {len(val)} '
            + f'validation instances, and {len(test)} test instances.')

    @classmethod
    def getDataset(self_class, config, seed=None):
        df = pd.read_csv(self_class.dataset_path, header=None)

        raw_data = df.values
        X = raw_data[:, 1:5].astype(float)
        Y = raw_data[:, 5]

        # one-hot encode the labels
        Y = pd.get_dummies(Y)

        X = torch.from_numpy(X)
        Y = torch.from_numpy(np.array(Y.values))

        data = torch.utils.data.TensorDataset(X, Y)

        train, val = torch.utils.data.random_split(data, [0.9, 0.1])

        train, test = torch.utils.data.random_split(train, [0.9, 0.1])

        train = self_class.subsetToDataset(train)
        val = self_class.subsetToDataset(val)
        test = self_class.subsetToDataset(test)

        return train, val, test

    @classmethod
    def getExampleInputs(self_class):
        train, val, test = self_class.getDataset(None)
        return train.tensors[0][:self_class.default_batch_size]
