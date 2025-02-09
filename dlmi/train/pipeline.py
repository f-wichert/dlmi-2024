from argparse import ArgumentParser

import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

from dlmi.data.dataset import CellSegmentationDataset
from dlmi.train.model.segmentation_model import SegmentationModel
from dlmi.train.model.unet import UNet
from dlmi.utils.utils import load_experiment_config


def main(config):
    max_epochs = config["train"]["max_epochs"]
    learning_rate = config["train"]["learning_rate"]
    if "count_weight" in config["train"]:
        count_weight = config["train"]["count_weight"]
    else:
        count_weight = 0.0

    image_patch_dir = config["dir"] / "train" / "prepared_images"
    mask_patch_dir = config["dir"] / "train" / "prepared_binary_mask"

    full_dataset = CellSegmentationDataset(
        image_dir=image_patch_dir, mask_dir=mask_patch_dir
    )
    if config["train"]["use_validation"]:
        val_size = config["train"]["validation_size"]
        train_size = 1 - config["train"]["validation_size"]
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size], generator
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["train"]["batch_size"],
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["train"]["batch_size"],
            shuffle=True,
        )

        monitor_loss = "val_dice"
        monitor_mode = "max"
    else:
        train_loader = DataLoader(
            full_dataset,
            batch_size=config["train"]["batch_size"],
            shuffle=True,
        )
        val_loader = None
        monitor_loss = "train_loss"
        monitor_mode = "min"

    model = UNet(
        in_channels=3,
        out_channels=2,
        depth=config["unet"]["depth"],
        pool_factor=config["unet"]["pool_factor"],
    )
    lit_model = SegmentationModel(model, learning_rate=learning_rate, count_weight=count_weight)

    # checkpoint_callback = ModelCheckpoint(
    #     monitor="train_loss",
    #     dirpath=ckpt_path,
    #     filename=f"checkpoint" + "-{epoch:02d}-{val_loss:.2f}",
    #     save_top_k=1,
    #     mode="min",
    # )
    best_model_callback = ModelCheckpoint(
        monitor=monitor_loss,
        dirpath=config["dir"] / "checkpoints",
        filename="best_model",
        save_top_k=1,
        mode=monitor_mode,
    )
    if config["logger"] == "wandb":
        logger = WandbLogger(project="dlmi", log_model=True)
        logger.log_hyperparams(
            {"monitor_loss": monitor_loss, "monitor_mode": monitor_mode, **config}
        )
    else:
        logger = TensorBoardLogger(save_dir="./tensorboard_logs")
    trainer = pl.Trainer(
        logger=logger,  # type: ignore
        max_epochs=max_epochs,
        callbacks=[best_model_callback],
        log_every_n_steps=1,
        accumulate_grad_batches=1,
    )

    trainer.fit(
        model=lit_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("experiment")
    args = parser.parse_args()
    exp = load_experiment_config(args.experiment)
    main(exp)
