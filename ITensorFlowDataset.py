from tffdataset.IDataset import IDataset

class ITensorFlowDataset(IDataset):
    def __init__(self, config, batch_size):
        super().__init__(config, batch_size)

    def batch(self):
        self.train = self.train.batch(self.batch_size)
        self.val = self.val.batch(self.batch_size)
        self.test = self.test.batch(self.batch_size)

    def trainSize(self):
        return self.train.cardinality().numpy()

    def valSize(self):
        return self.val.cardinality().numpy()

    def testSize(self):
        return self.test.cardinality().numpy()
