import torch
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
        # Estrazione parametro per evitare crash nel super().__init__
        no_object_weight_val = kwargs.pop('no_object_weight', 0.1)

        kwargs.setdefault('mask_coefficient', 2.0)
        kwargs.setdefault('dice_coefficient', 2.0)
        kwargs.setdefault('class_coefficient', 2.0)
        kwargs['no_object_coefficient'] = no_object_weight_val
        # Configurazione classi: 0=Sfondo, 1=Anomalia
        self.num_classes = 2
        self.stuff_classes = range(2)

        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=num_classes,
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            **kwargs
        )

        num_layers = self.network.num_blocks + 1 if getattr(self.network, 'masked_attn_enabled', False) else 1
        self.init_metrics_semantic(self.ignore_idx, num_layers)

        self.register_buffer("pixel_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _preprocess_images(self, imgs):
        if imgs.dtype == torch.uint8:
            imgs = imgs.float() / 255.0
        imgs = (imgs - self.pixel_mean) / self.pixel_std
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
            losses = self.criterion(
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
            # # --- FIX VISUALIZZAZIONE / EVAL ---
            # # Ignoriamo completamente la funzione to_per_pixel_logits_semantic del genitore
            # # che cerca di tagliare l'ultima classe. Facciamo il calcolo manualmente su 2 classi.
            #
            # # Creiamo logits binari: [Sfondo (-score), Anomalia (score)]
            # # Se score < 0 -> vince Sfondo
            # # Se score > 0 -> vince Anomalia
            # class_queries_logits = torch.cat(
            #     [-anomaly_logits, anomaly_logits], dim=-1
            # )  # [B, Q, 2]

            probs_anomaly = anomaly_logits.softmax(dim=-1)
            prob_bg_merged = probs_anomaly[..., 0] + probs_anomaly[..., 2]
            prob_anomaly = probs_anomaly[..., 1]
            valid_probs = torch.stack([prob_bg_merged, prob_anomaly], dim=-1)
            mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")  # [B, Q, H, W]

            # Calcolo manuale: Einsum (somma pesata delle maschere per probabilità classe)
            # Softmax su 2 classi assicura che P(Sfondo) + P(Anomalia) = 1.0. Non c'è Void.
            crop_logits = torch.einsum(
                "bqhw, bqc -> bchw",
                mask_logits.sigmoid(),
                valid_probs
            )

            logits = self.revert_window_logits_semantic(crop_logits, origins, img_sizes)

            self.update_metrics_semantic(logits, targets, i)

            if batch_idx == 0:
                self.plot_semantic(
                    imgs[0], targets[0], logits[0], log_prefix, i, batch_idx
                )
