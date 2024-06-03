import lightning as L
from typing import Optional
from torchvision.datasets import Kinetics
from torch.utils.data import DataLoader, random_split
import torch

class KineticsDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        Kinetics(root=self.data_dir, download=True, split='train', frames_per_clip=100, num_download_workers=10)
        Kinetics(root=self.data_dir, download=True, split='test', frames_per_clip=100, num_download_workers=10)

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            full_dataset = Kinetics(root=self.data_dir, download=False, split='train', frames_per_clip=100)
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
        if stage == 'test' or stage is None:
            self.test_dataset = Kinetics(root=self.data_dir, download=False, split='test', frames_per_clip=100)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)