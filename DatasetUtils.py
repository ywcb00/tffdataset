from enum import Enum

class DatasetID(Enum):
    FloodNet = 1
    Mnist = 2
    Iris = 3

def getDataset(config):
    match config["dataset_id"]:
        case DatasetID.FloodNet:
            from tffdataset.FloodNetDataset import FloodNetDataset
            return FloodNetDataset(config)
        case DatasetID.Mnist:
            from tffdataset.MnistDataset import MnistDataset
            return MnistDataset(config)
        case DatasetID.Iris:
            from tffdataset.IrisDataset import IrisDataset
            return IrisDataset(config)
        case _:
            raise NotImplementedError

def getDatasetElementSpec(config):
    match config["dataset_id"]:
        case DatasetID.FloodNet:
            from tffdataset.FloodNetDataset import FloodNetDataset
            return FloodNetDataset.element_spec
        case DatasetID.Mnist:
            from tffdataset.MnistDataset import MnistDataset
            return MnistDataset.element_spec
        case DatasetID.Iris:
            from tffdataset.IrisDataset import IrisDataset
            return IrisDataset.element_spec
        case _:
            raise NotImplementedError
