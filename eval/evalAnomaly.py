# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
import csv
import toml
import glob
import torch
import random
from PIL import Image
import numpy as np

from erfnet import ERFNet
import os.path as osp
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr, plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20
# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

input_transform = Compose(
    [
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
        # Normalize([.485, .456, .406], [.229, .224, .225]),
    ]
)

target_transform = Compose(
    [
        Resize((512, 1024), Image.NEAREST),
    ]
)


def select_device(cfg: dict) -> torch.device:
    device = cfg.get("device", "cpu").lower()

    match device:
        case "cpu":
            return torch.device("cpu")
        case "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        case "mps":
            return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        case _:
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return toml.load(f)


def main():

    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/eval_config.toml"
    cfg = load_config(config_path)

    device = select_device(cfg)

    print(f"Using device: {device}")

    out_root = cfg.get("outdir", "results_erfnet")
    os.makedirs(out_root, exist_ok=True)

    baselines = cfg.get("baselines", ["MSP", "MaxLogit", "MaxEntropy"])
    datasets = cfg.get("datasets", [])

    load_dir = cfg.get("load_dir", "../trained_models/")
    load_weights = cfg.get("load_weights", "erfnet_pretrained.pth")
    weightspath = osp.join(load_dir, load_weights)

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()

    file = open('results.txt', 'a')

    # print("Loading model: " + modelpath)
    print("Loading weights: " + weightspath)

    model = ERFNet(NUM_CLASSES)

    if device.type == 'cuda':
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    def load_my_state_dict(m, state_dict):
        own_state = m.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    continue
            else:
                own_state[name].copy_(param)
        return m

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print("Model and weights LOADED successfully")
    model.eval()

    # ERFNet CSV table
    ds_names = [d.get('table_name', d['name']) for d in datasets]
    csv_path = osp.join(out_root, f'erfnet_results.csv')
    table_rows: dict[str, dict[str, float]] = {b: {} for b in baselines}

    for d in datasets:
        ds_name = d["name"]
        table_name = d.get('table_name', ds_name)
        img_glob = d["images_glob"]

        if not (img_paths := sorted(glob.glob(img_glob))):
            print(f'No images found for dataset "{ds_name}" w/ glob: {img_glob}')
            continue

        for baseline in baselines:
            out_dir = osp.join(out_root, ds_name, baseline)
            os.makedirs(out_dir, exist_ok=True)

            print(f"Dataset: {ds_name:>20s} | Baseline: {baseline:>10s} | Images: {len(img_paths):>3d}", end=" => ")
            file.write(f"{ds_name} {baseline}\n")

            all_scores: list[np.ndarray] = []
            all_labels: list[np.ndarray] = []

            for path in img_paths:
                images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().to(device)
                with torch.no_grad():
                    logits = model(images)
                softmax_scores = torch.nn.functional.softmax(logits, dim=1)
                match baseline:
                    case "MSP":
                        anomaly_result = 1.0 - np.max(softmax_scores.squeeze(0).detach().cpu().numpy(), axis=0)
                    case "MaxLogit":
                        anomaly_result = -np.max(logits.squeeze(0).detach().cpu().numpy(), axis=0)
                    case "MaxEntropy":
                        p = softmax_scores.clamp_min(1e-12)
                        denom = torch.log(torch.tensor(float(p.shape[1]), device=p.device, dtype=p.dtype))
                        entropy = torch.sum(-p * torch.log(p), dim=1) / denom
                        anomaly_result = entropy.detach().cpu().numpy()[0].astype("float32")
                    case _:
                        raise ValueError(f"Unknown baseline: {baseline}")

                pathGT = path.replace("images", "labels_masks")

                if pathGT.endswith(".webp"):
                    pathGT = pathGT.replace("webp", "png")
                elif pathGT.endswith(".jpg"):
                    pathGT = pathGT.replace("jpg", "png")

                mask = Image.open(pathGT)
                mask = target_transform(mask)
                ood_gts = np.array(mask)

                if 'labels_masks' in pathGT:
                    ood_gts = np.where(ood_gts == 255, 255, (ood_gts > 0).astype(np.uint8))
                else:
                    # Dataset-specific remaps
                    if 'RoadAnomaly' in pathGT:
                        ood_gts = np.where((ood_gts == 2), 1, ood_gts)
                    if 'LostAndFound' in pathGT or 'LostFound' in pathGT:
                        ood_gts = np.where((ood_gts == 0), 255, ood_gts)
                        ood_gts = np.where((ood_gts == 1), 0, ood_gts)
                        ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)
                    if 'Streethazard' in pathGT:
                        ood_gts = np.where((ood_gts == 14), 255, ood_gts)
                        ood_gts = np.where((ood_gts < 20), 0, ood_gts)
                        ood_gts = np.where((ood_gts == 255), 1, ood_gts)

                # for FS Static, skip images with no anomaly pixels
                if 'fs_static' in pathGT.lower():
                    if 1 not in np.unique(ood_gts):
                        continue

                valid = (ood_gts != 255)
                if np.any(valid):
                    all_scores.append(anomaly_result[valid].astype(np.float32).reshape(-1))
                    all_labels.append((ood_gts[valid] == 1).astype(np.uint8).reshape(-1))

                base_name = osp.splitext(osp.basename(path))[0]
                save_path = osp.join(out_dir, f"{base_name}.npz")
                score = anomaly_result.astype(np.float16)

            if all_scores and all_labels:
                scores = np.concatenate(all_scores, axis=0)
                labels = np.concatenate(all_labels, axis=0)
                auprc = round(average_precision_score(labels, scores) * 100., 2)
                fpr95 = round(fpr_at_95_tpr(scores, labels) * 100., 2)
            else:
                auprc, fpr95 = np.nan, np.nan

            table_rows[baseline][f"{table_name}_AuPRC"] = auprc
            table_rows[baseline][f"{table_name}_FPR95"] = fpr95

            print(f"AUPRC: {auprc:5.2f}, FPR@TPR95: {fpr95:5.2f}")

    # Write ERFNet CSV table once, after all datasets/baselines
    headers = ["Model", "Method"]
    for name in ds_names:
        headers.extend([f"{name}_AuPRC", f"{name}_FPR95"])

    with open(csv_path, "w", newline='') as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=headers)
        writer.writeheader()
        for baseline in baselines:
            row = {"Model": "ERFNet", "Method": baseline}
            for name in ds_names:
                row[f"{name}_AuPRC"] = str(table_rows[baseline].get(f"{name}_AuPRC", 'nan'))
                row[f"{name}_FPR95"] = str(table_rows[baseline].get(f"{name}_FPR95", 'nan'))
            writer.writerow(row)

    file.write("\n")
    file.close()

    print(f"Saved per-image results under: {out_root}")
    print(f"Wrote ERFNet CSV table to: {csv_path}")


if __name__ == '__main__':
    main()
