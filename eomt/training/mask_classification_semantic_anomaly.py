from typing import List, Optional
import torch
from torch import nn

from .mask_classification_anomaly_loss import MaskClassificationAnomalyLoss
from .mask_classification_semantic import MaskClassificationSemantic

class MCS_Anomaly(MaskClassificationSemantic):

    def __init__(
        self, 
        network: nn.Module, 
        img_size: tuple[int, int], 
        num_classes: int,
        attn_mask_annealing_enabled: bool,
        stuff_classes: Optional[List[int]] = None,
        attn_mask_annealing_start_steps: Optional[List[int]] = None,
        attn_mask_annealing_end_steps: Optional[List[int]] = None,
        ignore_idx: int = 255,
        lr: float = 1e-4,
        llrd: float = 0.8,
        llrd_l2_enabled: bool = True,
        lr_mult: float = 1.0,
        weight_decay: float = 0.05,
        poly_power: float = 0.9,
        warmup_steps: List[int] = [500, 1000],
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        mask_coefficient: float = 5.0,
        dice_coefficient: float = 5.0,
        anomaly_coefficient: float = 2.0,
        ckpt_path: Optional[str] = None,
        delta_weights: bool = False,
        load_ckpt_class_head: bool = True,
    ):
        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=num_classes,
            stuff_classes=stuff_classes,
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            ignore_idx=ignore_idx,
            lr=lr,
            llrd=llrd,
            llrd_l2_enabled=llrd_l2_enabled,
            lr_mult=lr_mult,
            weight_decay=weight_decay,
            poly_power=poly_power,
            warmup_steps=warmup_steps,
            ckpt_path=None,  
            delta_weights=delta_weights,
            load_ckpt_class_head=load_ckpt_class_head,
        )
        
        # Load pretrained weights if a checkpoint path is provided
        if ckpt_path is not None:
            print(f"\n{'='*80}")
            print(f"Loading pretrained checkpoint from: {ckpt_path}")
            print(f"{'='*80}\n")
            
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            
            # Extract state_dict from checkpoint
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Filter only the network weights (exclude anomaly_head which does not exist in the pretrained model)
            network_state_dict = {}
            skipped_pos_embed = False
            
            for k, v in state_dict.items():
                if not k.startswith('network.'):
                    continue

                new_key = k.replace('network.', '', 1)

                if 'anomaly_head' in new_key:
                    continue
                if (not load_ckpt_class_head) and ('class_head' in new_key):
                    continue
                
                # Skip positional embeddings if shape mismatch (different image size)
                if 'pos_embed' in new_key or 'rope' in new_key:
                    # Check if shapes match
                    if new_key in self.network.state_dict():
                        current_shape = self.network.state_dict()[new_key].shape
                        if current_shape != v.shape:
                            print(f"⚠ Skipping {new_key}: shape mismatch (ckpt: {v.shape}, model: {current_shape})")
                            skipped_pos_embed = True
                            continue

                network_state_dict[new_key] = v
            
            # Load weights with strict=False to allow missing keys (anomaly_head)
            missing_keys, unexpected_keys = self.network.load_state_dict(network_state_dict, strict=False)
            
            print(f"✓ Successfully loaded pretrained weights")
            if skipped_pos_embed:
                print(f"✓ Positional embeddings re-initialized for current image size (640x640)")
            if missing_keys:
                anomaly_keys = [k for k in missing_keys if 'anomaly_head' in k]
                if anomaly_keys:
                    print(f"✓ Anomaly head keys not in checkpoint (expected): {anomaly_keys}")

                class_head_keys = []
                if not load_ckpt_class_head:
                    class_head_keys = [k for k in missing_keys if 'class_head' in k]
                    if class_head_keys:
                        print(f"✓ Class head keys intentionally skipped: {class_head_keys}")

                other_keys = [k for k in missing_keys if ('anomaly_head' not in k and (load_ckpt_class_head or 'class_head' not in k))]
                if other_keys:
                    print(f"⚠ Other missing keys: {other_keys}")
            if unexpected_keys:
                print(f"⚠ Unexpected keys in checkpoint: {unexpected_keys}")
            print(f"\n✓ Anomaly head initialized randomly and will be trained\n")

        # Initialize anomaly detection loss
        self.anomaly_criterion = MaskClassificationAnomalyLoss(
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            anomaly_coefficient=anomaly_coefficient,
        )

        # Freeze ALL network parameters
        print(f"{'='*80}")
        print("Freezing network parameters...")
        print(f"{'='*80}\n")
        
        for param in self.network.parameters():
            param.requires_grad = False

        # Unfreeze ONLY the anomaly_head
        if hasattr(self.network, 'anomaly_head') and self.network.anomaly_head is not None:
            for param in self.network.anomaly_head.parameters():
                param.requires_grad = True
            
            num_trainable = sum(p.numel() for p in self.network.anomaly_head.parameters())
            num_frozen = sum(p.numel() for p in self.network.parameters() if not p.requires_grad)
            
            print(f"✓ TRAINING Configuration:")
            print(f"  - Anomaly Head: TRAINABLE ({num_trainable:,} parameters)")
            print(f"  - Rest of Network: FROZEN ({num_frozen:,} parameters)")
            print(f"{'='*80}\n")
        else:
            raise ValueError("Network must have an anomaly_head for anomaly detection training. Use EoMT_EXT model.")

    def training_step(self, batch, batch_idx):
        imgs, targets = batch

        mask_logits_per_block, class_logits_per_block, anomaly_score_per_block = self(imgs)

        losses_all_blocks = {}
        for i, (mask_logits, class_logits, anomaly_scores) in enumerate(
            list(zip(mask_logits_per_block, class_logits_per_block, anomaly_score_per_block))
        ):
            # Compute anomaly detection loss using the criterion
            losses = self.anomaly_criterion(
                masks_queries_logits=mask_logits,
                anomaly_queries_logits=anomaly_scores,
                targets=targets,
            )
            
            block_postfix = self.block_postfix(i)
            losses = {f"{key}{block_postfix}": value for key, value in losses.items()}
            losses_all_blocks |= losses

        return self.anomaly_criterion.loss_total(losses_all_blocks, self.log)

    def eval_step(
        self,
        batch,
        batch_idx=None,
        log_prefix=None,
    ):
        """Override eval_step to handle anomaly detection (binary: normal vs anomaly)"""
        imgs, targets = batch

        img_sizes = [img.shape[-2:] for img in imgs]
        crops, origins = self.window_imgs_semantic(imgs)
        mask_logits_per_layer, class_logits_per_layer, anomaly_score_per_layer = self(crops)

        targets = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)

        for i, (mask_logits, class_logits, anomaly_scores) in enumerate(
            list(zip(mask_logits_per_layer, class_logits_per_layer, anomaly_score_per_layer))
        ):
            # Interpolate masks to image size
            mask_logits = nn.functional.interpolate(mask_logits, self.img_size, mode="bilinear")
            
            # Use anomaly scores instead of class logits for binary prediction
            # anomaly_scores: [B, Q, 1] -> sigmoid -> [B, Q]
            anomaly_probs = torch.sigmoid(anomaly_scores.squeeze(-1))  # [B, Q]
            
            # Create binary logits: [B, Q, 2] where [:,:,0]=normal, [:,:,1]=anomaly
            binary_class_logits = torch.stack([
                -anomaly_scores.squeeze(-1),  # logit for class 0 (normal)
                anomaly_scores.squeeze(-1)     # logit for class 1 (anomaly)
            ], dim=-1)  # [B, Q, 2]
            
            # Use semantic method but with binary class logits
            crop_logits = self.to_per_pixel_logits_semantic(mask_logits, binary_class_logits)
            logits = self.revert_window_logits_semantic(crop_logits, origins, img_sizes)

            self.update_metrics_semantic(logits, targets, i)

            if batch_idx == 0:
                self.plot_semantic(
                    imgs[0], targets[0], logits[0], log_prefix, i, batch_idx
                )

