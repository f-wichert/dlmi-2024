import lightning as pl
import torch
import torch.nn as nn
from torchmetrics.functional import dice
import torch.nn.functional as F


def find_connected_components_gpu(binary_mask):
    """
    Find connected components using GPU operations.
    Adapted from: https://gist.github.com/efirdc/5d8bd66859e574c683a504a4690ae8bc
    """
    device = binary_mask.device
    batch_size = binary_mask.shape[0]
    height, width = binary_mask.shape[1:]
    counts = torch.zeros(batch_size, device=device)

    labels = torch.arange(height * width, device=device).reshape((1, height, width))
    labels = labels.repeat(batch_size, 1, 1).float()

    labels = torch.where(binary_mask > 0, labels, torch.zeros_like(labels))

    max_iterations = 20  # Prevent infinite loops

    for b in range(batch_size):
        current_labels = labels[b].clone()
        prev_labels = torch.zeros_like(current_labels)
        iteration = 0
        while not torch.equal(current_labels, prev_labels) and iteration < max_iterations:
            prev_labels = current_labels.clone()

            pooled = F.max_pool2d(
                current_labels.unsqueeze(0),
                kernel_size=3,
                stride=1,
                padding=1
            ).squeeze(0)

            current_labels = torch.where(
                binary_mask[b] > 0,
                pooled,
                current_labels
            )

            iteration += 1

        counts[b] = torch.unique(current_labels[current_labels > 0]).numel()

    return counts


class CountingLoss(nn.Module):
    def __init__(self, base_criterion, count_weight=0.1):
        super().__init__()
        self.base_criterion = base_criterion
        self.count_weight = count_weight

    def forward(self, pred, target):
        seg_loss = self.base_criterion(pred, target)
        pred_binary = (pred > 0.5).float()
        target_binary = (target > 0.5).float()
        target_counts = find_connected_components_gpu(target_binary)
        pred_counts = find_connected_components_gpu(pred_binary[:, 1, :, :])
        count_loss = F.l1_loss(pred_counts, target_counts)

        return seg_loss + self.count_weight * count_loss, count_loss


class SegmentationModel(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3, count_weight=0.1):
        super(SegmentationModel, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self.cell_criteria = CountingLoss(self.criterion, count_weight=count_weight)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        preds = self(images)
        # loss = self.criterion(preds, masks)
        loss,  count_loss = self.cell_criteria(preds, masks)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_count_loss", count_loss, on_step=False, on_epoch=True, prog_bar=True)
        preds_classes = preds.argmax(dim=1)
        dice_score = dice(preds_classes, masks, num_classes=preds.shape[1])
        self.log("train_dice", dice_score, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        preds = self(images)
        # loss = self.criterion(preds, masks)
        loss, count_loss = self.cell_criteria(preds, masks)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_count_loss", count_loss, prog_bar=True)

        preds_classes = preds.argmax(dim=1)
        dice_score = dice(preds_classes, masks, num_classes=preds.shape[1])
        self.log("val_dice", dice_score, prog_bar=True)

    def test_step(self, batch, batch_idx):
        images, masks = batch
        preds = self(images)
        # loss = self.criterion(preds, masks)
        loss, count_loss = self.cell_criteria(preds, masks)
        self.log("test_loss", loss)

        preds_classes = preds.argmax(dim=1)
        dice_score = dice(preds_classes, masks, num_classes=preds.shape[1])
        self.log("test_dice", dice_score)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
