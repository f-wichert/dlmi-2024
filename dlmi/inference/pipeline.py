import os
from argparse import ArgumentParser

import torch
import numpy as np
from PIL import Image
from pathlib import Path

from patchify import unpatchify
from tqdm import tqdm

from dlmi.train.model.unet import UNet
from dlmi.train.model.segmentation_model import SegmentationModel
from dlmi.utils.utils import load_experiment_config


def load_model(checkpoint_path):
    """Load the trained model from checkpoint."""
    model = UNet(in_channels=3, out_channels=2)
    lit_model = SegmentationModel(model, learning_rate=0.001)

    checkpoint = torch.load(checkpoint_path)
    lit_model.load_state_dict(checkpoint['state_dict'])

    lit_model.eval()
    return lit_model


def preprocess_image(image_path):
    """Preprocess a single image for inference."""
    image = Image.open(image_path)
    image = np.array(image) / 255.0

    image = torch.from_numpy(image).float()

    if len(image.shape) == 2:  # Grayscale image
        image = image.unsqueeze(0).repeat(3, 1, 1)
    elif len(image.shape) == 3:  # RGB image
        image = image.permute(2, 0, 1)  # Change from HWC to CHW

    image = image.unsqueeze(0)

    if image.shape[-2:] != (128, 128):
        print(f"Warning: Resizing image from {image.shape[-2:]} to (128, 128)")
        image = torch.nn.functional.interpolate(image, size=(128, 128), mode='bilinear', align_corners=False)

    return image


def post_process_prediction(prediction):
    """Convert model output to binary segmentation mask."""
    pred = torch.argmax(prediction, dim=1)
    pred = pred.cpu().numpy()
    return pred


def save_prediction(prediction, output_path):
    """Save the prediction mask as an image."""
    # Scale to 0-255
    prediction = (prediction * 255).astype(np.uint8)
    # Save as image
    img = Image.fromarray(prediction[0])
    img.save(output_path)


def inference(model, test_dir, output_dir, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Perform inference on all images in test directory."""
    os.makedirs(output_dir, exist_ok=True)

    model = model.to(device)

    test_files = list(Path(test_dir).glob('*.png'))

    with torch.no_grad():
        for image_path in tqdm(test_files):
            image = preprocess_image(image_path)
            image = image.to(device)

            prediction = model(image)

            mask = post_process_prediction(prediction)

            output_path = Path(output_dir) / f"{image_path.stem}.png"
            save_prediction(mask, output_path)


def get_base_image_names(patch_filenames):
    """Extract unique base image names from patch filenames."""
    # Example: TCGA-GL-6846-01A-01-BS1_patch_bin_0_3.png -> TCGA-GL-6846-01A-01-BS1
    base_names = set()
    for filename in patch_filenames:
        base_name = filename.split('_patch_img_')[0]
        base_names.add(base_name)
    return list(base_names)


def get_patch_coordinates(filename, match="_patch_img_"):
    """Extract patch coordinates from filename."""
    # Example: TCGA-GL-6846-01A-01-BS1_patch_bin_0_3.png -> (0, 3)
    parts = filename.split(match)[1]
    x, y = parts.split('.')[0].split('_')
    return int(x), int(y)


def reconstruct_image(pred_dir_patches, pred_files, base_name, patch_size=(128, 128), step=128):
    """Reconstruct original image from patches."""
    patch_files = [
        os.path.join(pred_dir_patches, f) for f in pred_files
        if f.startswith(base_name)
    ]

    max_x = max(get_patch_coordinates(f)[0] for f in patch_files)
    max_y = max(get_patch_coordinates(f)[1] for f in patch_files)

    orig_height = (max_x + 1) * step
    orig_width = (max_y + 1) * step

    patches = np.zeros((max_x + 1, max_y + 1, patch_size[0], patch_size[1]))

    for patch_file in patch_files:
        x, y = get_patch_coordinates(patch_file)
        patch = np.array(Image.open(patch_file))
        patches[x, y] = patch

    return unpatchify(patches, (orig_width, orig_height))


def main(config):
    if config["test"]["use_best_model"]:
        checkpoint_path = config["dir"] / "checkpoints" / "best_model.ckpt"
    else:
        checkpoint_path = config["dir"] / "checkpoints" / config["test"]["model_name"]

    image_patch_dir = config["dir"] / "test" / "prepared_images"
    pred_dir = config["dir"] / "test" / "predictions"
    pred_dir_patches = str(pred_dir) + "_patches"
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(pred_dir_patches, exist_ok=True)

    model = load_model(checkpoint_path)

    inference(model, image_patch_dir, pred_dir_patches)

    print("Inference completed. Results saved in:", pred_dir_patches)
    pred_files = [f for f in os.listdir(pred_dir_patches) if f.endswith('.png')]

    base_names = get_base_image_names(pred_files)

    print(f"Found {len(base_names)} original images to reconstruct")

    if config["data"]["test"]["patchify"]:
        for base_name in tqdm(base_names, desc="Reconstructing images"):
            reconstructed = reconstruct_image(pred_dir_patches, pred_files, base_name)
            output_path = os.path.join(pred_dir, f"{base_name}.png")
            Image.fromarray(reconstructed.astype(np.uint8)).save(output_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("experiment")
    args = parser.parse_args()
    exp = load_experiment_config(args.experiment)
    main(exp)
