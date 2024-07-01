from dataset.IDataset import IDataset

import logging
import numpy as np
import tensorflow as tf

class MnistDataset(IDataset):
    def __init__(self, config):
        super().__init__(config, batch_size=30)
        self.logger = logging.getLogger("dataset/MnistDataset")
        self.logger.setLevel(config["log_level"])

    def load(self):
        train, val, test = self.getDataset(self.config)
        self.train = train
        self.val = val
        self.test = test
        self.logger.info(f'Found {train.cardinality().numpy()} train instances, {val.cardinality().numpy()} '
            + f'validation instances, and {test.cardinality().numpy()} test instances.')

    @classmethod
    def getDataset(self_class, config):
        num_classes = 10

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # scale the image values to [0, 1]
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255

        # FIXME: this is only a work-around for the following error:
        #   serialized size of the federated train dataset exceeds maximum allowed for federated training process
        x_train = x_train[1:30000]
        y_train = y_train[1:30000]

        # expand the image dimensions from (28, 28) to (28, 28, 1)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

        # one-hot encode the labels
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

        # concatenate images with labels
        train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        # split train dataset into train and validation set
        train, val = tf.keras.utils.split_dataset(train, left_size=0.9, right_size=None,
            shuffle=True, seed=config["seed"])

        return train, val, test
