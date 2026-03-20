from enum import Enum

class DatasetID(Enum):
    FloodNet = 1
    Mnist = 2
    Iris = 3
    IrisPyTorch = 4

def getDataset(config):
    return getDatasetClass(config)(config)

def getDatasetClass(config):
    match config["dataset_id"]:
        case DatasetID.FloodNet:
            from tffdataset.FloodNetDataset import FloodNetDataset
            return FloodNetDataset
        case DatasetID.Mnist:
            from tffdataset.MnistDataset import MnistDataset
            return MnistDataset
        case DatasetID.Iris:
            from tffdataset.IrisDataset import IrisDataset
            return IrisDataset
        case DatasetID.IrisPyTorch:
            from tffdataset.IrisPyTorchDataset import IrisPyTorchDataset
            return IrisPyTorchDataset
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
        case DatasetID.IrisPyTorch:
            from tffdataset.IrisPyTorchDataset import IrisPyTorchDataset
            return IrisPyTorchDataset.element_spec
        case _:
            raise NotImplementedError
