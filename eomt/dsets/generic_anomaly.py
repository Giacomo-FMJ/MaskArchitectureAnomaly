import os
import glob
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, PILToTensor, ToTensor

IGNORE_INDEX = 255


def normalize_gt(gt: np.ndarray, gt_format: str) -> np.ndarray:
    """
    Normalize a GT mask to:
      0 = ID/normal
      1 = OOD/anomaly
      255 = void/ignore
    Returns uint8 array HxW.
    """

    if gt.dtype != np.uint8:
        gt = gt.astype(np.uint8)

    fmt = gt_format.lower()

    if fmt == "binary_any_nonzero_is_ood_void255":
        # 255 stays void, any other non-zero becomes OOD
        return np.where(gt == IGNORE_INDEX, IGNORE_INDEX, (gt > 0).astype(np.uint8)).astype(np.uint8)

    if fmt == "binary_255_is_ood":
        # FS Static: 255 means anomaly, everything else normal
        return (gt == IGNORE_INDEX).astype(np.uint8)

    raise ValueError(f"Unknown gt_format='{gt_format}'")


class GenericAnomalyDataset(Dataset):
    """
    Class representing a generic anomaly dataset.

    This class provides functionality to handle a dataset for anomaly detection,
    including image and mask loading, transformations, and creating targets for
    model input. The dataset expects an organized directory structure to locate
    images and corresponding masks. It supports filtering images without any out-of-distribution
    anomalies if configured. The primary goal is to prepare images and their associated
    ground truth annotations for training or evaluating anomaly detection models.

    :ivar cfg: Dictionary containing dataset configuration parameters.
    :type cfg: dict
    :ivar name: Name of the dataset, derived from the configuration.
    :type name: str
    :ivar root: Root directory where the dataset is stored.
    :type root: str
    :ivar images_dir: Subdirectory under the root containing images.
    :type images_dir: str
    :ivar masks_dir: Subdirectory under the root containing masks or labels.
    :type masks_dir: str
    :ivar image_glob: Glob pattern to match image files.
    :type image_glob: str
    :ivar mask_ext: File extension for the mask files.
    :type mask_ext: str
    :ivar gt_format: Ground truth format used in the dataset (used for normalization).
    :type gt_format: Any
    :ivar skip_no_ood: Boolean flag indicating whether to skip images without out-of-distribution anomalies.
    :type skip_no_ood: bool
    :ivar img_size: Tuple specifying the image size for resizing operations.
    :type img_size: tuple
    :ivar img_transform: Transformation pipeline for input images, including resizing and tensor conversion.
    :type img_transform: torchvision.transforms.Compose
    :ivar mask_resize: Transformation pipeline for resizing masks.
    :type mask_resize: torchvision.transforms.Resize
    :ivar img_paths: List of file paths for valid images in the dataset.
    :type img_paths: list
    """
    def __init__(self, ds_cfg: dict, img_size=(1024, 1024)):

        self.cfg = ds_cfg
        self.name = ds_cfg["name"]
        self.root = ds_cfg["root"]
        self.images_dir = ds_cfg.get("images_dir", "images")
        self.masks_dir = ds_cfg.get("masks_dir", "labels_masks")
        self.image_glob = ds_cfg.get("image_glob", "*")
        self.mask_ext = ds_cfg.get("mask_ext", "png")
        self.gt_format = ds_cfg["gt_format"]
        self.skip_no_ood = bool(ds_cfg.get("skip_no_ood", False))
        self.img_size = tuple(img_size)

        self.img_transform = Compose([Resize(self.img_size, Image.BILINEAR), PILToTensor()])
        self.mask_resize = Resize(self.img_size, Image.NEAREST)

        pattern = os.path.join(self.root, self.images_dir, self.image_glob)
        self.img_paths = sorted(glob.glob(pattern))
        if not self.img_paths:
            raise FileNotFoundError(f"[{self.name}] No images found with glob: {pattern}")

        if self.skip_no_ood:
            self.img_paths = [p for p in self.img_paths if self._has_any_ood(p)]

    def __len__(self):
        return len(self.img_paths)

    def _img_to_mask_path(self, img_path: str) -> str:
        rel = os.path.relpath(img_path, self.root)
        rel = rel.replace(self.images_dir, self.masks_dir, 1)
        base, _ = os.path.splitext(rel)
        return os.path.join(self.root, base + "." + self.mask_ext)

    def _has_any_ood(self, img_path: str) -> bool:
        mpath = self._img_to_mask_path(img_path)
        if not os.path.exists(mpath):
            return False
        m = Image.open(mpath)
        m = self.mask_resize(m)
        gt = normalize_gt(np.array(m, dtype=np.uint8), self.gt_format)
        return np.any(gt == 1)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        mask_path = self._img_to_mask_path(img_path)

        img = Image.open(img_path).convert("RGB")
        img_t = self.img_transform(img)  # [3,H,W] uint8 (PilTensor)

        mask = Image.open(mask_path)
        mask = self.mask_resize(mask)
        gt = normalize_gt(np.array(mask, dtype=np.uint8), self.gt_format)  # {0,1,255}
        gt_t = torch.from_numpy(gt.astype(np.int64))  # [H,W] long

        # Two classes: 0 = ID/normal, 1 = OOD/anomaly. Pixels with 255 are ignored by not belonging to any mask
        m0 = (gt_t == 0)
        m1 = (gt_t == 1)
        masks = torch.stack([m0, m1], dim=0)  # [2,H,W] bool
        labels = torch.tensor([0, 1], dtype=torch.int64)
        target = {"masks": masks, "labels": labels}

        return img_t, target
