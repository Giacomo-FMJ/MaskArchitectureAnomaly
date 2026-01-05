import torch
import torch._dynamo
from torch import nn
import torch.nn.functional as F
from typing import Optional, List

from .mask_classification_loss import MaskClassificationLoss
from .mask_classification_semantic import MaskClassificationSemantic


class MCS_Anomaly(MaskClassificationSemantic):

    def __init__(
            self,
            network: nn.Module,
            img_size: tuple[int, int],
            num_classes: int,
            attn_mask_annealing_enabled: bool,
            attn_mask_annealing_start_steps: Optional[List[int]] = None,
            attn_mask_annealing_end_steps: Optional[List[int]] = None,
            **kwargs
    ):
        num_classes_model = 1

        if attn_mask_annealing_enabled:
            n_blocks = getattr(network, 'num_blocks', 3)
            default_len = n_blocks + 2
            if attn_mask_annealing_start_steps is None:
                attn_mask_annealing_start_steps = [0] * default_len
            if attn_mask_annealing_end_steps is None:
                attn_mask_annealing_end_steps = [0] * default_len

        # Coefficienti ottimizzati per stabilità
        kwargs.setdefault('mask_coefficient', 2.0)
        kwargs.setdefault('dice_coefficient', 2.0)
        kwargs.setdefault('class_coefficient', 2.0)

        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=num_classes_model,
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            **kwargs
        )

        self.num_classes = 2
        self.stuff_classes = range(2)
        num_layers = self.network.num_blocks + 1 if getattr(self.network, 'masked_attn_enabled', False) else 1
        self.init_metrics_semantic(self.ignore_idx, num_layers)

        for param in self.network.parameters():
            param.requires_grad = True

        if hasattr(self.network, 'encoder'):
            print("Freezing Encoder (Backbone) parameters.")
            for param in self.network.encoder.parameters():
                param.requires_grad = False
        else:
            print("Warning: 'encoder' attribute not found. Backbone might not be frozen!")

        self.register_buffer("pixel_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _preprocess_images(self, imgs):
        """Normalizza le immagini per la backbone (ImageNet stats)."""
        if imgs.dtype == torch.uint8:
            imgs = imgs.float() / 255.0
        imgs = (imgs - self.pixel_mean) / self.pixel_std
        return imgs

    def forward(self, x):
        # FIX CRITICO: Applichiamo la normalizzazione qui!
        # Senza questo, la rete riceve raw pixels e non impara nulla.
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

        # self(imgs) ora chiama forward() che normalizza
        mask_logits_per_block, class_logits_per_block, anomaly_score_per_block = self(imgs)

        filtered_targets = []
        for t in targets:
            anomaly_mask = t["masks"][1]
            if anomaly_mask.sum() > 0:
                filtered_targets.append({
                    "masks": anomaly_mask.unsqueeze(0).float(),
                    "labels": torch.zeros(1, dtype=torch.long, device=t["labels"].device)
                })
            else:
                filtered_targets.append({
                    "masks": torch.zeros((0, *anomaly_mask.shape[-2:]), dtype=torch.float,
                                         device=anomaly_mask.device),
                    "labels": torch.zeros(0, dtype=torch.long, device=t["labels"].device)
                })

        losses_all_blocks = {}
        for i, (mask_logits, class_logits, anomaly_scores) in enumerate(
                list(zip(mask_logits_per_block, class_logits_per_block, anomaly_score_per_block))
        ):
            # FIX: [score, 0] allinea la scala con eval_step e stabilizza i gradienti
            class_queries_logits = torch.cat(
                [anomaly_scores, torch.zeros_like(anomaly_scores)], dim=-1
            )

            losses = self.criterion(
                masks_queries_logits=mask_logits,
                class_queries_logits=class_queries_logits,
                targets=filtered_targets,
            )
            block_postfix = self.block_postfix(i)
            losses = {f"{key}{block_postfix}": value for key, value in losses.items()}
            losses_all_blocks |= losses

        return self.criterion.loss_total(losses_all_blocks, self.log)

    def eval_step(self, batch, batch_idx=None, log_prefix=None):
        imgs, targets = batch
        targets = self._unstack_targets(imgs, targets)
        img_sizes = [img.shape[-2:] for img in imgs]

        # window_imgs_semantic usa PIL/uint8, quindi passiamo imgs grezze
        crops, origins = self.window_imgs_semantic(imgs)

        # self(crops) normalizzerà internamente grazie al nuovo forward()
        mask_logits_per_layer, class_logits_per_layer, anomaly_scores_per_layer = self(crops)

        targets = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)

        for i, (mask_logits, class_logits, anomaly_scores) in enumerate(
                list(zip(mask_logits_per_layer, class_logits_per_layer, anomaly_scores_per_layer))
        ):
            # Eval: [0, score].
            # Se score > 0 -> Class 1 (Anomaly) vince.
            # Se score < 0 -> Class 0 (Background) vince.
            class_queries_logits = torch.cat(
                [torch.zeros_like(anomaly_scores), anomaly_scores], dim=-1
            )

            mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")
            crop_logits = self.to_per_pixel_logits_semantic(mask_logits, class_queries_logits)
            logits = self.revert_window_logits_semantic(crop_logits, origins, img_sizes)

            self.update_metrics_semantic(logits, targets, i)

            if batch_idx == 0:
                self.plot_semantic(
                    imgs[0], targets[0], logits[0], log_prefix, i, batch_idx
                )