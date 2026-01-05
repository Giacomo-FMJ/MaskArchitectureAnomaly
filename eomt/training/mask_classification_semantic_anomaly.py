from torch import nn

from .mask_classification_loss import MaskClassificationLoss
from .mask_classification_semantic import MaskClassificationSemantic

class MCS_Anomaly(MaskClassificationSemantic):

    def __init__(self, network: nn.Module, img_size: tuple[int, int], num_classes: int,
                 attn_mask_annealing_enabled: bool):

        super().__init__(network, img_size, num_classes, attn_mask_annealing_enabled)


        #freeze della class head
        if hasattr(network, "class_head"):
            for param in network.class_head:
                param.requires_grad = False


        #ensure unfreeze of new head
        if hasattr(self.network, 'anomaly_head'):
            for param in self.network.anomaly_head.parameters():
                param.requires_grad = True



    def training_step(self, batch, batch_idx):
        imgs, targets = batch

        mask_logits_per_block, class_logits_per_block, anomaly_score_per_block = self(imgs)

        losses_all_blocks = {}
        for i, (mask_logits, class_logits, anomaly_scores) in enumerate(
            list(zip(mask_logits_per_block, class_logits_per_block, anomaly_score_per_block))
        ):
            losses = self.criterion(
                masks_queries_logits=mask_logits,
                class_queries_logits=class_logits,
                targets=targets,
            )
            block_postfix = self.block_postfix(i)
            losses = {f"{key}{block_postfix}": value for key, value in losses.items()}
            losses_all_blocks |= losses

        return self.criterion.loss_total(losses_all_blocks, self.log)


