from tffdataset.IDataset import IDataset

import csv
import logging
import os
import pandas as pd
from pathlib import Path
import tensorflow as tf

class IrisDataset(IDataset):
    num_classes = 3 # either flooded or not flooded
    element_spec = (tf.TensorSpec(shape=(None, 3000, 3000, 3), dtype=tf.float32,),
        tf.TensorSpec(shape=(None,), dtype=tf.bool))
    element_spec = (tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 3), dtype=tf.float32))

    dataset_path = Path(r'./data/iris.csv')
    dataset_url = r'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'

    def __init__(self, config):
        super().__init__(config, batch_size=2)
        self.logger = logging.getLogger("dataset/IrisDataset")
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
        self.logger.info(f'Found {train.cardinality().numpy()} train instances, {val.cardinality().numpy()} '
            + f'validation instances, and {test.cardinality().numpy()} test instances.')

    @classmethod
    def getDataset(self_class, config, seed=None):
        df = pd.read_csv(self_class.dataset_path, header=None)

        raw_data = df.values
        X = raw_data[:, 1:5].astype(float)
        Y = raw_data[:, 5]

        X = tf.convert_to_tensor(X)
        Y = tf.convert_to_tensor(Y)

        # one-hot encode the labels
        Y = pd.get_dummies(Y)

        data = tf.data.Dataset.from_tensor_slices((X, Y))

        train, test = tf.keras.utils.split_dataset(data, left_size=0.9, right_size=None,
            shuffle=True, seed=seed)

        train, val = tf.keras.utils.split_dataset(train, left_size=0.9, right_size=None,
            shuffle=True, seed=seed)

        return train, val, test
