from dataset.FloodNetDataset import FloodNetDataset
from dataset.MnistDataset import MnistDataset

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
