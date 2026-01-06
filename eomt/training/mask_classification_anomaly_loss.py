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
)


class MaskClassificationAnomalyLoss(nn.Module):
    """
    Loss for anomaly detection using mask queries.
    Handles 3 classes: void (255, ignored), background (0), anomaly (1).
    
    Uses Hungarian matching to assign queries to ground truth masks,
    then computes binary cross-entropy on anomaly scores.
    """
    
    def __init__(
        self,
        num_points: int,
        oversample_ratio: float,
        importance_sample_ratio: float,
        mask_coefficient: float,
        dice_coefficient: float,
        anomaly_coefficient: float,
    ):
        super().__init__()
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.mask_coefficient = mask_coefficient
        self.dice_coefficient = dice_coefficient
        self.anomaly_coefficient = anomaly_coefficient

        # Use Hungarian matcher for query-to-GT assignment
        self.matcher = Mask2FormerHungarianMatcher(
            num_points=num_points,
            cost_mask=mask_coefficient,
            cost_dice=dice_coefficient,
            cost_class=0.0,  # No class matching, only mask-based matching
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

        # Dummy class logits for matcher (cost_class=0 so they're ignored)
        # Just need the right shape [B, Q, num_classes]
        batch_size, num_queries = masks_queries_logits.shape[:2]
        dummy_class_logits = torch.zeros(
            batch_size, num_queries, 2, 
            device=masks_queries_logits.device,
            dtype=masks_queries_logits.dtype
        )

        # Use Hungarian matching ONLY on masks (cost_class=0)
        indices = self.matcher(
            masks_queries_logits=masks_queries_logits,
            mask_labels=mask_labels,
            class_queries_logits=dummy_class_logits,  # Ignored due to cost_class=0
            class_labels=class_labels,
        )

        # Compute anomaly classification loss
        loss_anomaly = self.loss_anomaly_labels(
            anomaly_queries_logits, class_labels, indices
        )

        return loss_anomaly

    def loss_anomaly_labels(
        self,
        anomaly_queries_logits: torch.Tensor,
        class_labels: List[torch.Tensor],
        indices: List[tuple],
    ):
        """
        Compute binary cross-entropy loss for anomaly detection.
        
        Args:
            anomaly_queries_logits: [B, Q, 1] - anomaly logits per query
            class_labels: list of [N] tensors with GT labels {0, 1}
            indices: list of (query_idx, gt_idx) tuples from Hungarian matching
        
        Returns:
            dict with 'anomaly_cross_entropy' loss
        """
        batch_size = anomaly_queries_logits.shape[0]
        num_queries = anomaly_queries_logits.shape[1]
        device = anomaly_queries_logits.device

        # Prepare target labels for each query
        target_labels_per_query = torch.zeros(
            batch_size, num_queries, dtype=torch.float32, device=device
        )

        for b, (query_idx, gt_idx) in enumerate(indices):
            # query_idx: indices of queries matched to GT
            # gt_idx: indices of GT masks matched to queries
            # Assign GT labels to matched queries
            target_labels_per_query[b, query_idx] = class_labels[b][gt_idx].float()

        # Flatten for loss computation
        anomaly_logits_flat = anomaly_queries_logits.squeeze(-1)  # [B, Q]
        
        # Compute binary cross-entropy loss
        loss = F.binary_cross_entropy_with_logits(
            anomaly_logits_flat.view(-1),
            target_labels_per_query.view(-1),
            reduction='mean'
        )

        # Normalize by number of masks
        num_masks = sum(len(gt_idx) for (_, gt_idx) in indices)
        num_masks_tensor = torch.as_tensor(
            num_masks, dtype=torch.float, device=device
        )

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_masks_tensor)
            world_size = dist.get_world_size()
        else:
            world_size = 1

        num_masks = torch.clamp(num_masks_tensor / world_size, min=1)
        
        # Scale by batch size since BCE already averages
        loss = loss * (batch_size * num_queries / num_masks)

        return {"anomaly_cross_entropy": loss}

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

            if "anomaly" in loss_key:
                weighted_loss = loss * self.anomaly_coefficient
            else:
                raise ValueError(f"Unknown loss key: {loss_key}")

            if loss_total is None:
                loss_total = weighted_loss
            else:
                loss_total = torch.add(loss_total, weighted_loss)

        log_fn("losses/train_loss_total", loss_total, sync_dist=True, prog_bar=True)

        return loss_total
