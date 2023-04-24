from Data.Data import NiftiData
from torch.utils.data import DataLoader, random_split,  ConcatDataset
from pytorch_lightning import LightningDataModule

class MRIdata(LightningDataModule):
    def __init__(self, batch_size: int = None, SWIN_size: tuple[int, int, int] = None):
        super().__init__()
        scans = NiftiData(SWIN_size)

        self.train, self.val = random_split(scans, [int(len(scans) * 0.8), len(scans) - int(len(scans) * 0.8)], )
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=12, drop_last=True, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=12, drop_last=True, pin_memory=True, persistent_workers=True)
