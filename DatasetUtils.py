from tffdataset.FloodNetDataset import FloodNetDataset
from tffdataset.MnistDataset import MnistDataset

from enum import Enum

class DatasetID(Enum):
    FloodNet = 1
    Mnist = 2

def getDataset(config):
    match config["dataset_id"]:
        case DatasetID.FloodNet:
            return FloodNetDataset(config)
        case DatasetID.Mnist:
            return MnistDataset(config)
        case _:
            raise NotImplementedError

def getDatasetElementSpec(config):
    match config["dataset_id"]:
        case DatasetID.FloodNet:
            return FloodNetDataset.element_spec
        case DatasetID.Mnist:
            return MnistDataset.element_spec
        case _:
            raise NotImplementedError
