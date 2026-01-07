# ---------------------------------------------------------------
# Minimal Lightning DataModule wrapping GenericAnomalyDataset
# ---------------------------------------------------------------
from typing import Optional, Sequence

from torch.utils.data import DataLoader
import torch

from .lightning_data_module import LightningDataModule
from .generic_anomaly import GenericAnomalyDataset

class AnomalyDataModule(LightningDataModule):
    def __init__(
        self,
        ds_cfg: dict,
        batch_size: int = 4,
        num_workers: int = 4,
        img_size: tuple[int, int] = (640, 640),
        num_classes: int = 2,
        stuff_classes: Optional[Sequence[int]] = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ) -> None:
        super().__init__(
            path=ds_cfg.get("root", ""),
            batch_size=batch_size,
            num_workers=num_workers,
            img_size=tuple(img_size),
            num_classes=num_classes,
            check_empty_targets=False,
            ignore_idx=ds_cfg.get("ignore_idx", 255),
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        self.ds_cfg = ds_cfg
        self.num_classes = num_classes
        self.stuff_classes = list(stuff_classes) if stuff_classes is not None else list(range(num_classes))
        self.dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        # Single dataset used for train/val/test (semantic segmentation approach)
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
