from Data.Data import NiftiData
from torch.utils.data import DataLoader, random_split,  ConcatDataset
from pytorch_lightning import LightningDataModule

class MRIdata(LightningDataModule):
    def __init__(self, batch_size: int = None, img_size: int = None, dimensions:int = None):
        super().__init__()
        scans = NiftiData()

        self.train, self.val = random_split(scans, [int(len(scans) * 0.8), len(scans) - int(len(scans) * 0.8)], )
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=5, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=5, drop_last=True)

