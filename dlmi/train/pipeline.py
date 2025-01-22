import os
from argparse import ArgumentParser

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from dlmi.data.dataset import CellSegmentationDataset
from dlmi.utils.utils import load_config, load_experiment_config
from dlmi.train.model.segmentation_model import SegmentationModel
from dlmi.train.model.unet import UNet


def main(config):
    max_epochs = config["train"]["max_epochs"]
    learning_rate = config["train"]["learning_rate"]

    image_patch_dir = config["dir"] / "train" / "prepared_images"
    mask_patch_dir = config["dir"] / "train" / "prepared_binary_mask"

    dataset = CellSegmentationDataset(image_dir=image_patch_dir, mask_dir=mask_patch_dir)

    train_loader = DataLoader(dataset, batch_size=config["train"]["batch_size"], shuffle=True)

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
    parser = ArgumentParser()
    parser.add_argument("experiment")
    args = parser.parse_args()
    exp = load_experiment_config(args.experiment)
    main(exp)

