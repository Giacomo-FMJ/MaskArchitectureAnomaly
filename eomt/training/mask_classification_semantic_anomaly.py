import torch
from torch import nn
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from typing import Optional, List

from .mask_classification_loss import MaskClassificationLoss
from .mask_classification_semantic import MaskClassificationSemantic


class MCS_Anomaly(MaskClassificationSemantic):

    def __init__(
            self,
            network: nn.Module,
            img_size: tuple[int, int],
            attn_mask_annealing_enabled: bool,
            attn_mask_annealing_start_steps: Optional[List[int]] = None,
            attn_mask_annealing_end_steps: Optional[List[int]] = None,
            num_points: int = 12544,
            num_classes: int = 19,
            ignore_idx: int = 255,
            lr: float = 1e-5,
            llrd: float = 0.8,
            llrd_l2_enabled: bool = True,
            lr_mult: float = 1.0,
            weight_decay: float = 0.05,
            oversample_ratio: float = 3.0,
            importance_sample_ratio: float = 0.75,
            poly_power: float = 0.9,
            warmup_steps: List[int] = [500, 1000],
            mask_coefficient: float = 2.0,
            dice_coefficient: float = 2.0,
            class_coefficient: float = 2.0,
            mask_thresh: float = 0.8,
            overlap_thresh: float = 0.8,
            ckpt_path: Optional[str] = None,
            delta_weights: bool = False,
            load_ckpt_class_head: bool = True,
            **kwargs
    ):

        no_object_coefficient = 1.0

        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=num_classes,
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            ignore_idx=ignore_idx,
            lr=lr,
            llrd=llrd,
            llrd_l2_enabled=llrd_l2_enabled,
            lr_mult=lr_mult,
            weight_decay=weight_decay,
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            poly_power=poly_power,
            warmup_steps=warmup_steps,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            mask_thresh=mask_thresh,
            overlap_thresh=overlap_thresh,
            ckpt_path=ckpt_path,
            delta_weights=delta_weights,
            load_ckpt_class_head=load_ckpt_class_head,
            no_object_coefficient=no_object_coefficient,
            **kwargs
        )

        self.criterion_anomalymask = MaskClassificationLoss(
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            num_labels=2,
            no_object_coefficient=no_object_coefficient,
        )

        num_layers = self.network.num_blocks + 1 if getattr(self.network, 'masked_attn_enabled', False) else 1
        self.init_metrics_semantic(self.ignore_idx, num_layers)

        self.register_buffer("pixel_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        for param in self.network.parameters():
            param.requires_grad = False

        if hasattr(self.network, 'anomaly_head'):
            print("Unfreezing anomaly_head params...")
            for param in self.network.anomaly_head.parameters():
                param.requires_grad = True

        if hasattr(self.network, 'mask_head'):
            print("Unfreezing mask_head params...")
            for param in self.network.mask_head.parameters():
                param.requires_grad = True

    def _preprocess_images(self, imgs):
        if imgs.dtype == torch.uint8:
            imgs = imgs.float() / 255.0
        #imgs = (imgs - self.pixel_mean) / self.pixel_std
        return imgs

    def forward(self, x):
        x = self._preprocess_images(x)
        return self.network(x)

    def _unstack_targets(self, imgs, targets):
        if isinstance(targets, dict):
            batch_size = imgs.shape[0]
            return [{k: v[i] for k, v in targets.items()} for i in range(batch_size)]
        return targets

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        targets = self._unstack_targets(imgs, targets)

        mask_logits_per_block, class_logits_per_block, anomaly_logits_per_block = self(imgs)

        losses_all_blocks = {}
        for i, (mask_logits, class_logits, anomaly_logits) in enumerate(
                list(zip(mask_logits_per_block, class_logits_per_block, anomaly_logits_per_block))
        ):
            losses = self.criterion_anomalymask(
                masks_queries_logits=mask_logits,
                class_queries_logits=anomaly_logits,
                targets=targets,
            )
            block_postfix = self.block_postfix(i)
            losses = {f"{key}{block_postfix}": value for key, value in losses.items()}
            losses_all_blocks |= losses

        return self.criterion.loss_total(losses_all_blocks, self.log)

    def eval_step(self, batch, batch_idx=None, log_prefix=None):
        imgs, targets = batch
        targets = self._unstack_targets(imgs, targets)
        img_sizes = [img.shape[-2:] for img in imgs]

        crops, origins = self.window_imgs_semantic(imgs)

        mask_logits_per_layer, class_logits_per_layer, anomaly_logits_per_layer = self(crops)

        targets = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)

        for i, (mask_logits, class_logits, anomaly_logits) in enumerate(
                list(zip(mask_logits_per_layer, class_logits_per_layer, anomaly_logits_per_layer))
        ):
            # probs_anomaly = anomaly_logits.softmax(dim=-1)
            #
            # # --- FIX LOGICA CLASSI (Sfondo+Void=0, Anomalia=1) ---
            # valid_probs = torch.stack([
            #     probs_anomaly[..., 0] + probs_anomaly[..., 2],  # Canale 0: Normal (Bg + Void)
            #     probs_anomaly[..., 1]  # Canale 1: Anomaly
            # ], dim=-1)

            mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")

            # # Einsum corretto: [B, Q, H, W] * [B, Q, C] -> [B, C, H, W]
            # crop_logits = torch.einsum(
            #     "bqhw, bqc -> bchw",
            #     mask_logits.sigmoid(),
            #     valid_probs
            # )
            crop_logits = self.to_per_pixel_logits_semantic(mask_logits, anomaly_logits)

            logits = self.revert_window_logits_semantic(crop_logits, origins, img_sizes)

            self.update_metrics_semantic(logits, targets, i)

            if batch_idx == 0:
                self.plot_semantic(
                    imgs[0], targets[0], logits[0], log_prefix, i, batch_idx
                )

    def plot_semantic(self, img, target, logits, prefix, layer_idx, batch_idx):
        import wandb

        # 1. Immagine Input (Raw)
        # Se l'immagine è uint8 (0-255), la portiamo a float 0-1.
        # Se è già float, ci assicuriamo che sia nel range 0-1.
        if img.dtype == torch.uint8:
            img_vis = img.float() / 255.0
        else:
            img_vis = img.clone().cpu()
            # Se per qualche motivo arriva in range 0-255 float
            if img_vis.max() > 1.1:
                img_vis = img_vis / 255.0

        img_vis = torch.clamp(img_vis, 0, 1).cpu()

        # 2. Predizione (Probabilità)
        # logits shape: [NumClasses(3 o 2), H, W]. Indice 1 = Anomalia.
        # Applichiamo Softmax per avere probabilità valide 0..1
        #print(logits.shape)  #2x1024x1024
        sum_scores = logits.sum(dim=0, keepdim=True) + 1e-6
        probs = (logits / sum_scores).cpu()
        prob_vis = probs[1] #anomaly probs
        pred_vis = prob_vis.unsqueeze(0).repeat(3, 1, 1)
        pred_vis = torch.clamp(pred_vis, 0, 1)

        # 3. Ground Truth (BG=0, Void=100, Anomaly=255)
        target_vis = target.clone().cpu()
        vis_t = torch.zeros_like(pred_vis)  # [3, H, W]

        # Creiamo un canale unico prima
        vis_map = torch.zeros_like(target_vis, dtype=torch.float32)

        # Mappa: Anomalia (1) -> Bianco (1.0)
        vis_map[target_vis == 1] = 1.0
        # Mappa: Void (255) -> Grigio (100/255 ~= 0.39)
        vis_map[target_vis == self.ignore_idx] = 100.0 / 255.0

        # Replica su 3 canali per RGB
        vis_t = vis_map.unsqueeze(0).repeat(3, 1, 1)

        # 4. Combina: [Input Originale | Ground Truth | Predizione Probabilità]
        comparison = torch.cat([img_vis, vis_t, pred_vis], dim=2)

        # 5. Log su WandB
        if hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'log'):
            caption = f"{prefix}_L{layer_idx}_Input_GT_PredProb"
            self.logger.experiment.log({
                f"val_images/{prefix}_layer_{layer_idx}": [
                    wandb.Image(comparison, caption=caption)
                ]
            })

        # Debug locale (opzionale)
        if batch_idx == 0 and layer_idx == 0:
            save_image(comparison, f"debug_vis_{prefix}.png")
