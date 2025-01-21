import os

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from dlmi.data.dataset import CellSegmentationDataset
from dlmi.shared.config import load_config
from dlmi.train.model.segmentation_model import SegmentationModel
from dlmi.train.model.unet import UNet


def main():
    config = load_config()
    max_epochs = config["train"]["max_epochs"]
    learning_rate = config["train"]["learning_rate"]

    image_patch_dir = os.path.join(
        config["data"]["data_dir"],
        config["data"]["train"]["dir"],
        config["data"]["train"]["augment_after"]["patch"]["image_dir"],
    )
    mask_patch_dir = os.path.join(
        config["data"]["data_dir"],
        config["data"]["train"]["dir"],
        config["data"]["train"]["augment_after"]["patch"]["binary_dir"],
    )

    dataset = CellSegmentationDataset(
        image_dir=image_patch_dir, mask_dir=mask_patch_dir
    )

    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = UNet(in_channels=3, out_channels=2)
    lit_model = SegmentationModel(model, learning_rate=learning_rate)

    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        dirpath="checkpoints/",
        filename=f"checkpoint" + "-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )
    logger = TensorBoardLogger(save_dir="./tensorboard_logs")

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
    )

    trainer.fit(
        model=lit_model,
        train_dataloaders=train_loader,
    )


if __name__ == "__main__":
    main()

