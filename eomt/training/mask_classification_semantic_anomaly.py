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
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        mask_coefficient: float = 5.0,
        dice_coefficient: float = 5.0,
        anomaly_coefficient: float = 2.0,
    ):

        super().__init__(network, img_size, num_classes, attn_mask_annealing_enabled)

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
        for param in self.network.parameters():
            param.requires_grad = False

        # Unfreeze ONLY the anomaly_head
        if hasattr(self.network, 'anomaly_head') and self.network.anomaly_head is not None:
            for param in self.network.anomaly_head.parameters():
                param.requires_grad = True
        else:
            raise ValueError("Network must have an anomaly_head for anomaly detection training")

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

