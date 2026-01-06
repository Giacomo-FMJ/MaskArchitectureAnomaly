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

        # Unpacking dei 3 output [Mask, Class, Anomaly(Bg, Ano, Void)]
        mask_logits_per_layer, class_logits_per_layer, anomaly_logits_per_layer = self(crops)

        targets = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)

        for i, (mask_logits, class_logits, anomaly_logits) in enumerate(
                list(zip(mask_logits_per_layer, class_logits_per_layer, anomaly_logits_per_layer))
        ):
            mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")  # [B, Q, H, W]

            # 1. Softmax su tutti e 3 i canali (Bg, Ano, Void)
            # anomaly_logits ha shape [B, Q, 3]
            probs = anomaly_logits.softmax(dim=-1)

            # 2. Logica di soppressione Void
            # Se è Void (idx 2), contribuisce alla certezza di Background (idx 0).
            # Void = "Nessun Oggetto" = Sicuramente non Anomalia = Background
            valid_probs = probs.clone()
            valid_probs[..., 0] = valid_probs[..., 0] + valid_probs[..., 2]

            # 3. Tronchiamo a 2 classi [Background, Anomaly]
            valid_probs = valid_probs[..., :2]

            # 4. Calcolo Probability Map per pixel (Soft Segmentation)
            # range [0, 1]
            crop_probs = torch.einsum(
                "bqhw, bqc -> bchw",
                mask_logits.sigmoid(),
                valid_probs
            )

            # 5. Ricostruzione (Stitching) nello spazio delle probabilità
            # Più stabile mediare probabilità tra 0 e 1 che logit grezzi
            final_probs = self.revert_window_logits_semantic(crop_probs, origins, img_sizes)

            # 6. Pulizia NaN post-stitching
            if torch.isnan(final_probs).any():
                final_probs = torch.nan_to_num(final_probs, nan=0.0)

            # 7. Conversione finale in Pseudo-Logits
            # Le metriche si aspettano logits per fare la loro softmax/argmax interna o cross-entropy
            # Usiamo clamp(min=1e-8) per evitare il -inf del log(0)
            logits = torch.log(final_probs.clamp(min=1e-8))

            self.update_metrics_semantic(logits, targets, i)

            if batch_idx == 0:
                self.plot_semantic(
                    imgs[0], targets[0], logits[0], log_prefix, i, batch_idx
                )