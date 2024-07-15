from tffdataset.IDataset import IDataset

import logging

class DirectDataset(IDataset):
    def __init__(self, batch_size, element_spec, train, val, test, config):
        super().__init__(config, batch_size=batch_size)
        self.logger = logging.getLogger("dataset/DirectDataset")
        self.logger.setLevel(config["log_level"])
        DirectDataset.element_spec = element_spec
        self.train = train
        self.val = val
        self.test = test
        self.logger.info(f'Directly instantiated a dataset with '
            + f'{train.cardinality().numpy()} train instances, {val.cardinality().numpy()} '
            + f'validation instances, and {test.cardinality().numpy()} test instances.')

    def load(self):
        self.logger.error("Cannot load a directly instantiated dataset.")
        raise RuntimeError
