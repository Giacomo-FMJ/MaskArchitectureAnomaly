import os
import glob
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, PILToTensor
from .lightning_data_module import LightningDataModule

IGNORE_INDEX = 255


def normalize_gt(gt: np.ndarray, gt_format: str) -> np.ndarray:
    if gt.dtype != np.uint8:
        gt = gt.astype(np.uint8)
    fmt = gt_format.lower()
    if fmt == "binary_any_nonzero_is_ood_void255":
        return np.where(gt == IGNORE_INDEX, IGNORE_INDEX, (gt > 0).astype(np.uint8)).astype(np.uint8)
    if fmt == "binary_255_is_ood":
        return (gt == IGNORE_INDEX).astype(np.uint8)
    raise ValueError(f"Unknown gt_format='{gt_format}'")


class AnomalyInternalDataset(Dataset):
    """
    Classe Dataset pura che gestisce il caricamento dei dati.
    Questa classe è sicura da serializzare (pickle) per i worker su Windows.
    """

    def __init__(self, datasets_cfgs: list, img_size: tuple[int, int]):
        self.datasets_cfgs = datasets_cfgs
        self.img_size = tuple(img_size)

        # Trasformazioni
        self.img_transform = Compose([Resize(self.img_size, Image.BILINEAR), PILToTensor()])
        self.mask_resize = Resize(self.img_size, Image.NEAREST)

        self.samples = []
        for ds_cfg in self.datasets_cfgs:
            self._load_dataset(ds_cfg)

        if not self.samples:
            print("Warning: No images found in any of the provided datasets.")

    def _load_dataset(self, ds_cfg: dict):
        root = ds_cfg["root"]
        images_dir = ds_cfg.get("images_dir", "images")
        masks_dir = ds_cfg.get("masks_dir", "labels_masks")
        image_glob = ds_cfg.get("image_glob", "*")
        mask_ext = ds_cfg.get("mask_ext", "png")
        gt_format = ds_cfg["gt_format"]
        skip_no_ood = bool(ds_cfg.get("skip_no_ood", False))

        pattern = os.path.join(root, images_dir, image_glob)
        img_paths = sorted(glob.glob(pattern))

        if not img_paths:
            print(f"Warning: No images found for {ds_cfg['name']} at {pattern}")
            return

        for img_path in img_paths:
            rel = os.path.relpath(img_path, root)
            rel = rel.replace(images_dir, masks_dir, 1)
            base, _ = os.path.splitext(rel)
            mask_path = os.path.join(root, base + "." + mask_ext)

            if skip_no_ood:
                if not self._check_has_ood(mask_path, gt_format):
                    continue

            self.samples.append((img_path, mask_path, gt_format))

    def _check_has_ood(self, mask_path: str, gt_format: str) -> bool:
        if not os.path.exists(mask_path):
            return False
        try:
            m = Image.open(mask_path)
            m = self.mask_resize(m)
            gt = normalize_gt(np.array(m, dtype=np.uint8), gt_format)
            return np.any(gt == 1)
        except Exception:
            return False

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path, gt_format = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        img_t = self.img_transform(img)

        if os.path.exists(mask_path):
            mask = Image.open(mask_path)
            mask = self.mask_resize(mask)
            gt = normalize_gt(np.array(mask, dtype=np.uint8), gt_format)
        else:
            gt = np.full(self.img_size, IGNORE_INDEX, dtype=np.uint8)

        gt_t = torch.from_numpy(gt.astype(np.int64))

        present_classes = torch.unique(gt_t)
        masks_list = []
        labels_list = []

        # Check for class 0 (Backround) and 1 (Anomaly) presence
        for class_id in [0, 1]:
            if class_id in present_classes:
                masks_list.append(gt_t == class_id)
                labels_list.append(class_id)

        if masks_list:
            masks = torch.stack(masks_list, dim=0)
            labels = torch.tensor(labels_list, dtype=torch.int64)
        else:
            # Handle case where neither 0 nor 1 is present (e.g. all 255)
            masks = torch.zeros((0, *gt_t.shape), dtype=torch.bool)
            labels = torch.tensor([], dtype=torch.int64)

        target = {"masks": masks, "labels": labels}

        return img_t, target


class GenericAnomalyDataset(LightningDataModule):
    """
    LightningDataModule che istanzia e avvolge AnomalyInternalDataset.
    """

    def __init__(
            self,
            datasets: list,
            img_size: tuple[int, int] = (1024, 1024),
            batch_size: int = 4,
            num_workers: int = 4,
            path: str = "",
            num_classes: int = 19,
            check_empty_targets: bool = False
    ):
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            img_size=img_size,
            num_classes=num_classes,
            check_empty_targets=check_empty_targets
        )

        self.batch_size = batch_size
        self.num_workers = num_workers

        # Istanziamo il dataset interno qui.
        # Questo oggetto è più semplice e sicuro da passare ai worker.
        self.dataset = AnomalyInternalDataset(datasets, img_size)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )
