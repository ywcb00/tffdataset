from dataset.IDataset import IDataset

import csv
import logging
import os
import tensorflow as tf

class FloodNetDataset(IDataset):
    def __init__(self, config):
        super().__init__(config, batch_size=2)
        self.logger = logging.getLogger("dataset/FloodNetDataset")
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
        train_dir = "./data/FloodNet-Supervised_v1.0/train/train-org-img/"
        val_dir = "./data/FloodNet-Supervised_v1.0/val/val-org-img/"
        test_dir = "./data/FloodNet-Supervised_v1.0/test/test-org-img/"
        # load the image dataset
        train = tf.keras.utils.image_dataset_from_directory(
            directory = train_dir,
            label_mode = None,
            batch_size = None,
            image_size = (3000, 3000),
            shuffle = False,
            color_mode = "rgb")
        val = tf.keras.utils.image_dataset_from_directory(
            directory = val_dir,
            label_mode = None,
            batch_size = None,
            image_size = (3000, 3000),
            shuffle = False,
            color_mode = "rgb")
        test = tf.keras.utils.image_dataset_from_directory(
            directory = test_dir,
            label_mode = None,
            batch_size = None,
            image_size = (3000, 3000),
            shuffle = False,
            color_mode = "rgb")

        # obtain the filenames
        train_fnames = map(lambda fp: os.path.splitext(os.path.basename(fp))[0],
            train.file_paths)
        val_fnames = map(lambda fp: os.path.splitext(os.path.basename(fp))[0],
            val.file_paths)
        test_fnames = map(lambda fp: os.path.splitext(os.path.basename(fp))[0],
            test.file_paths)

        # read/compute the reponse
        train_response = dict()
        val_response = dict()
        test_response = dict()
        if(not config["force_load"] and os.path.exists(config["train_response_path"])
            and os.path.exists(config["val_response_path"])
            and os.path.exists(config["test_response_path"])):
            train_response, val_response, test_response = self_class.readResponse(config)
        else:
            train_response, val_response, test_response = self_class.loadResponse(config)

        # create list of responses in the same order as the image dataset
        train_labels = [train_response[fname] for fname in train_fnames]
        val_labels = [val_response[fname] for fname in val_fnames]
        test_labels = [test_response[fname] for fname in test_fnames]

        # add the responseS to the dataset
        train = tf.data.Dataset.zip((train,
            tf.data.Dataset.from_tensor_slices(tf.constant(train_labels), name="response")))
        val = tf.data.Dataset.zip((val,
            tf.data.Dataset.from_tensor_slices(tf.constant(val_labels), name="response")))
        test = tf.data.Dataset.zip((test,
            tf.data.Dataset.from_tensor_slices(tf.constant(test_labels), name="response")))

        # TODO: reduced dataset only for development purpose
        train = train.take(250)
        val = val.take(100)
        test = test.take(40)

        return train, val, test

    @classmethod
    def loadResponse(self_class, config):
        def isFlooded(img):
            counts = tf.map_fn(
                fn = lambda fc: tf.reduce_sum(tf.cast(tf.equal(tf.cast(
                    tf.round(tf.reshape(img, [-1])), tf.uint32), fc), tf.uint32)),
                elems = config["flooded_classes"])
            flooded_count = tf.reduce_sum(counts)
            # return True if the count of flooded pixels exceeds the specified threshold
            return tf.greater(flooded_count, tf.cast(tf.multiply(tf.cast(tf.size(img), tf.float32),
                    config["flooded_threshold"]), tf.uint32))

        # read the response label images from disk
        train_labels_dir = "./data/FloodNet-Supervised_v1.0/train/train-label-img/"
        val_labels_dir = "./data/FloodNet-Supervised_v1.0/val/val-label-img/"
        test_labels_dir = "./data/FloodNet-Supervised_v1.0/test/test-label-img/"
        train_labels = tf.keras.utils.image_dataset_from_directory(
            directory = train_labels_dir,
            label_mode = None,
            batch_size = None,
            image_size = (3000, 3000),
            shuffle = False,
            color_mode = "grayscale")
        val_labels = tf.keras.utils.image_dataset_from_directory(
            directory = val_labels_dir,
            label_mode = None,
            batch_size = None,
            image_size = (3000, 3000),
            shuffle = False,
            color_mode = "grayscale")
        test_labels = tf.keras.utils.image_dataset_from_directory(
            directory = test_labels_dir,
            label_mode = None,
            batch_size = None,
            image_size = (3000, 3000),
            shuffle = False,
            color_mode = "grayscale")

        train_fnames = map(lambda fp: os.path.basename(fp).split("_")[0],
            train_labels.file_paths)
        val_fnames = map(lambda fp: os.path.basename(fp).split("_")[0],
            val_labels.file_paths)
        test_fnames = map(lambda fp: os.path.basename(fp).split("_")[0],
            test_labels.file_paths)

        # create the response (True/False) from the label images
        train_response = train_labels.map(isFlooded)
        val_response = val_labels.map(isFlooded)
        test_response = test_labels.map(isFlooded)

        train_response = map(lambda tb: tb.numpy(), train_labels.map(isFlooded))
        val_response = map(lambda tb: tb.numpy(), val_labels.map(isFlooded))
        test_response = map(lambda tb: tb.numpy(), test_labels.map(isFlooded))

        train_response = dict(zip(train_fnames, train_response))
        val_response = dict(zip(val_fnames, val_response))
        test_response = dict(zip(test_fnames, test_response))

        # write the computed binary responses to disk for later usage
        with open(config["train_response_path"], "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["fname", "response"])
            for key, val in train_response.items():
                writer.writerow([key, val])
        with open(config["val_response_path"], "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["fname", "response"])
            for key, val in val_response.items():
                writer.writerow([key, val])
        with open(config["test_response_path"], "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["fname", "response"])
            for key, val in test_response.items():
                writer.writerow([key, val])

        return train_response, val_response, test_response

    @classmethod
    def readResponse(self_class, config):
        train_response = dict()
        with open(config["train_response_path"], "r") as csvfile:
            reader = csv.reader(csvfile)
            next(reader) # omit the header
            for row in reader:
                train_response[row[0]] = (row[1] == "True")
        val_response = dict()
        with open(config["val_response_path"], "r") as csvfile:
            reader = csv.reader(csvfile)
            next(reader) # omit the header
            for row in reader:
                val_response[row[0]] = (row[1] == "True")
        test_response = dict()
        with open(config["test_response_path"], "r") as csvfile:
            reader = csv.reader(csvfile)
            next(reader) # omit the header
            for row in reader:
                test_response[row[0]] = (row[1] == "True")

        return train_response, val_response, test_response
