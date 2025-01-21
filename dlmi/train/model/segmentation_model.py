import os

import lightning as pl
import torch
import torch.nn as nn
from torchmetrics.functional import dice


class SegmentationModel(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3):
        super(SegmentationModel, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        preds = self(images)
        loss = self.criterion(preds, masks)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        preds_classes = preds.argmax(dim=1)
        dice_score = dice(preds_classes, masks, num_classes=preds.shape[1])
        self.log("train_dice", dice_score, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # def validation_step(self, batch, batch_idx):
    #     images, masks = batch
    #     preds = self(images)
    #     loss = self.criterion(preds, masks)
    #     self.log("val_loss", loss, prog_bar=True)
    #
    #     preds_classes = preds.argmax(dim=1)
    #     dice_score = dice(preds_classes, masks, num_classes=preds.shape[1])
    #     self.log("val_dice", dice_score, prog_bar=True)
    #
    # def test_step(self, batch, batch_idx):
    #     images, masks = batch
    #     preds = self(images)
    #     loss = self.criterion(preds, masks)
    #     self.log("test_loss", loss)
    #
    #     preds_classes = preds.argmax(dim=1)
    #     dice_score = dice(preds_classes, masks, num_classes=preds.shape[1])
    #     self.log("test_dice", dice_score)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
