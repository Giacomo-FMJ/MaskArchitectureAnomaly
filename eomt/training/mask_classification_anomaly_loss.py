# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the Hugging Face Transformers library,
# specifically from the Mask2Former loss implementation, which itself is based on
# Mask2Former and DETR by Facebook, Inc. and its affiliates.
# Used under the Apache 2.0 License.
# ---------------------------------------------------------------

from typing import List
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerHungarianMatcher,
    Mask2FormerLoss,
)


class MaskClassificationAnomalyLoss(Mask2FormerLoss):
    """
    Loss for anomaly detection using mask queries.
    Handles 3 classes: void (255, ignored), background (0), anomaly (1).
    
    Uses Hungarian matching to assign queries to ground truth masks,
    then computes:
    1. Binary cross-entropy on anomaly scores (with class weighting)
    2. Mask loss (binary cross-entropy on mask logits)
    3. Dice loss (for mask quality)
    """
    
    def __init__(
        self,
        num_points: int,
        oversample_ratio: float,
        importance_sample_ratio: float,
        mask_coefficient: float,
        dice_coefficient: float,
        anomaly_coefficient: float,
        anomaly_weight: float = 2.0,  # Weight for anomaly class (higher = more importance)
        background_weight: float = 0.1,  # Weight for background class (lower to prevent dominance)
    ):
        # Initialize as nn.Module (not calling super().__init__() to avoid Mask2FormerLoss init)
        nn.Module.__init__(self)
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.mask_coefficient = mask_coefficient
        self.dice_coefficient = dice_coefficient
        self.anomaly_coefficient = anomaly_coefficient
        self.anomaly_weight = anomaly_weight
        self.background_weight = background_weight

        # Use Hungarian matcher with class cost for better anomaly matching
        self.matcher = Mask2FormerHungarianMatcher(
            num_points=num_points,
            cost_mask=mask_coefficient,
            cost_dice=dice_coefficient,
            cost_class=anomaly_coefficient,  # Use class cost for anomaly matching
        )

    @torch.compiler.disable
    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        anomaly_queries_logits: torch.Tensor,
        targets: List[dict],
    ):
        """
        Compute anomaly detection loss.
        
        Args:
            masks_queries_logits: [B, Q, H, W] - mask logits per query
            anomaly_queries_logits: [B, Q, 1] - anomaly score per query (logit)
            targets: list of dicts with 'masks' [N, H, W] and 'labels' [N]
                     labels are {0: background, 1: anomaly}
        
        Returns:
            dict with loss components
        """
        mask_labels = [
            target["masks"].to(masks_queries_logits.dtype) for target in targets
        ]
        class_labels = [target["labels"].long() for target in targets]

        # Convert anomaly logits to class logits [B, Q, 2] for matcher
        # class 0 = background, class 1 = anomaly
        batch_size, num_queries = masks_queries_logits.shape[:2]
        class_logits = torch.zeros(
            batch_size, num_queries, 2, 
            device=masks_queries_logits.device,
            dtype=masks_queries_logits.dtype
        )
        class_logits[:, :, 0] = -anomaly_queries_logits.squeeze(-1)  # logit for background
        class_logits[:, :, 1] = anomaly_queries_logits.squeeze(-1)   # logit for anomaly

        # Use Hungarian matching with mask + class costs
        indices = self.matcher(
            masks_queries_logits=masks_queries_logits,
            mask_labels=mask_labels,
            class_queries_logits=class_logits,
            class_labels=class_labels,
        )

        # Compute all three losses
        loss_masks = self.loss_masks(masks_queries_logits, mask_labels, indices)
        loss_anomaly = self.loss_anomaly_labels(
            anomaly_queries_logits, class_labels, indices
        )

        return {**loss_masks, **loss_anomaly}

    def loss_masks(
        self,
        masks_queries_logits: torch.Tensor,
        mask_labels: List[torch.Tensor],
        indices: List[tuple],
    ):
        """
        Compute mask loss (BCE) and dice loss for mask refinement.
        Reuses Mask2Former's implementation for consistency.
        """
        # Use parent class (Mask2FormerLoss) implementation
        losses = super().loss_masks(
            masks_queries_logits, mask_labels, indices, num_masks=1
        )
        
        # Normalize by number of masks across all GPUs
        num_masks = sum(len(tgt) for (_, tgt) in indices)
        num_masks_tensor = torch.as_tensor(
            num_masks, dtype=torch.float, device=masks_queries_logits.device
        )

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_masks_tensor)
            world_size = dist.get_world_size()
        else:
            world_size = 1

        num_masks = torch.clamp(num_masks_tensor / world_size, min=1)

        for key in losses.keys():
            losses[key] = losses[key] / num_masks

        return losses

    def loss_anomaly_labels(
        self,
        anomaly_queries_logits: torch.Tensor,
        class_labels: List[torch.Tensor],
        indices: List[tuple],
    ):
        """
        Compute Weighted Binary Cross-Entropy.
        Args:
            anomaly_queries_logits: [B, Q, 1] - anomaly score logits per query
            class_labels: list of [N] tensors with ground truth labels (0=background, 1=anomaly, 255=void)
            indices: list of tuples with matched (query_idx, gt_idx) per batch item
        """
        batch_size, num_queries = anomaly_queries_logits.shape[:2]
        device = anomaly_queries_logits.device

        # 1. Initialize Targets and Weights
        # Default Target = 0 (Background)
        target_labels = torch.zeros(batch_size, num_queries, dtype=torch.float32, device=device)
        
        # Default Weight = 0.1 (Down-weight background to prevent dominance)
        query_weights = torch.ones(batch_size, num_queries, device=device) * self.background_weight

        # 2. Assign Targets based on Matching
        for b, (query_idx, gt_idx) in enumerate(indices):
            query_idx = query_idx.to(device)
            gt_idx = gt_idx.to(device)
            
            # Get the actual labels for matched GT masks
            matched_labels = class_labels[b][gt_idx]
            
            # Filter out VOID (255) labels
            valid_mask = matched_labels != 255
            valid_query_idx = query_idx[valid_mask]
            valid_labels = matched_labels[valid_mask]
            
            # Set targets based on actual labels (0 for background, 1 for anomaly)
            target_labels[b, valid_query_idx] = valid_labels.float()
            
            # Set higher weight only for anomaly queries
            anomaly_mask = valid_labels == 1
            if anomaly_mask.any():
                anomaly_query_idx = valid_query_idx[anomaly_mask]
                query_weights[b, anomaly_query_idx] = self.anomaly_weight

        # Flatten for BCE
        logits_flat = anomaly_queries_logits.view(-1)
        targets_flat = target_labels.view(-1)
        weights_flat = query_weights.view(-1)

        # 3. Compute Weighted BCE
        loss = F.binary_cross_entropy_with_logits(
            logits_flat,
            targets_flat,
            weight=weights_flat, # Applies the per-element weighting
            reduction='sum'
        )

        # 4. Normalize
        # Normalize by the actual number of anomaly masks to keep loss magnitude stable
        num_masks = sum(len(gt_idx) for (_, gt_idx) in indices)
        num_masks_tensor = torch.as_tensor(num_masks, dtype=torch.float, device=device)

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_masks_tensor)
            num_masks = torch.clamp(num_masks_tensor / dist.get_world_size(), min=1)
        else:
            num_masks = max(num_masks_tensor.item(), 1)

        return {"anomaly_cross_entropy": loss / num_masks}

    def loss_total(self, losses_all_layers, log_fn) -> torch.Tensor:
        """
        Aggregate losses from all layers.
        
        Args:
            losses_all_layers: dict of {loss_name: loss_value}
            log_fn: function to log metrics
        
        Returns:
            total weighted loss
        """
        loss_total = None
        for loss_key, loss in losses_all_layers.items():
            log_fn(f"losses/train_{loss_key}", loss, sync_dist=True)

            # Apply appropriate weighting for each loss type
            if "mask" in loss_key:
                weighted_loss = loss * self.mask_coefficient
            elif "dice" in loss_key:
                weighted_loss = loss * self.dice_coefficient
            elif "anomaly" in loss_key:
                weighted_loss = loss * self.anomaly_coefficient
            else:
                raise ValueError(f"Unknown loss key: {loss_key}")

            if loss_total is None:
                loss_total = weighted_loss
            else:
                loss_total = torch.add(loss_total, weighted_loss)

        log_fn("losses/train_loss_total", loss_total, sync_dist=True, prog_bar=True)

        return loss_total
