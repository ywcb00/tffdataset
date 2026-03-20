from tffdataset.ITensorFlowDataset import ITensorFlowDataset

import logging

class DirectTensorFlowDataset(ITensorFlowDataset):
    def __init__(self, batch_size, element_spec, train, val, test, config):
        super().__init__(config, batch_size=batch_size)
        self.logger = logging.getLogger("dataset/DirectTensorFlowDataset")
        self.logger.setLevel(config["log_level"])
        DirectTensorFlowDataset.element_spec = element_spec
        self.train = train
        self.val = val
        self.test = test
        self.logger.info(f'Directly instantiated a dataset with '
            + f'{len(train)} train instances, {len(val)} '
            + f'validation instances, and {len(test)} test instances.')

    def load(self):
        self.logger.error("Cannot load a directly instantiated dataset.")
        raise RuntimeError
