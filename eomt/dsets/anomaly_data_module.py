# ---------------------------------------------------------------
# Minimal Lightning DataModule wrapping GenericAnomalyDataset
# ---------------------------------------------------------------
from typing import Optional

import lightning
from torch.utils.data import DataLoader
import torch

from .generic_anomaly import GenericAnomalyDataset


class AnomalyDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        ds_cfg: dict,
        batch_size: int = 4,
        num_workers: int = 4,
        img_size: tuple[int, int] = (640, 640),
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ) -> None:
        super().__init__()
        self.ds_cfg = ds_cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = tuple(img_size)
        self.dataloader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": False if num_workers == 0 else persistent_workers,
        }
        self.dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        # Single dataset used for train/val/test; change here if you want splits
        self.dataset = GenericAnomalyDataset(ds_cfg=self.ds_cfg, img_size=self.img_size)

    def train_dataloader(self) -> DataLoader:
        if self.dataset is None:
            self.setup("fit")
        return DataLoader(self.dataset, shuffle=True, collate_fn=self.train_collate, **self.dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        if self.dataset is None:
            self.setup("fit")
        return DataLoader(self.dataset, shuffle=False, collate_fn=self.train_collate, **self.dataloader_kwargs)

    def test_dataloader(self) -> DataLoader:
        if self.dataset is None:
            self.setup("test")
        return DataLoader(self.dataset, shuffle=False, collate_fn=self.eval_collate, **self.dataloader_kwargs)

    @staticmethod
    def train_collate(batch):
        imgs, targets = [], []
        for img, target in batch:
            imgs.append(img)
            targets.append(target)
        return torch.stack(imgs), targets

    @staticmethod
    def eval_collate(batch):
        return tuple(zip(*batch))
