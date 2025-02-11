import logging
import os
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from patchify import patchify
from PIL import Image, ImageDraw
from skimage.draw import polygon, disk
from skimage.morphology import binary_dilation
from tqdm import tqdm

from dlmi.utils.utils import load_experiment_config

logger = logging.getLogger(__name__)


def augment_random(file_pairs, method_config, data_dir, cell_info):
    torch.manual_seed(42)
    augmented_files = []
    bin_dir = data_dir / "prepared_binary_mask"
    img_dir = data_dir / "prepared_images"
    os.makedirs(bin_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    for img_path, img, bin_mask, color_mask, xml_path in tqdm(file_pairs, desc="Augmenting data"):
        base_name = Path(img_path).stem
        img_size_x, img_size_y = img.shape[:2]
        aug_methods = method_config["methods"]

        for i in range(method_config["num_augmentations"]):
            new_img_path = img_dir / f"{base_name}_aug_{i}"
            new_bin_path = bin_dir / f"{base_name}_aug_{i}"
            transformations = []
            if "random_crop" in aug_methods:
                transformations.append(transforms.RandomCrop(size=(img_size_x * 0.8, img_size_y * 0.8)))
            if "random_rotation" in aug_methods:
                transformations.append(transforms.RandomRotation(degrees=(15, 75)))
            if "random_horizontal_flip" in aug_methods:
                transformations.append(transforms.RandomHorizontalFlip())
            if "random_vertical_flip" in aug_methods:
                transformations.append(transforms.RandomVerticalFlip())
            if "gaussian_blur" in aug_methods:
                transformations.append(transforms.GaussianBlur(kernel_size=3))
            if "elastic_transform" in aug_methods:
                transformations.append(transforms.ElasticTransform(alpha=8.0))

            if "pick_random" in method_config:
                transformations = np.random.choice(transformations, method_config["pick_random"], replace=False)
                transform = transforms.Compose(transformations)
            else:
                transform = transforms.Compose(transformations)

            if "color_jitter" in method_config and method_config["color_jitter"]:
                color_transform = transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.1,
                    hue=0.02
                )
            else:
                color_transform = None

            # Convert numpy arrays to PIL Images with correct modes
            img_pil = Image.fromarray(img.astype('uint8'), 'RGB')  # For color images
            bin_mask_pil = Image.fromarray(bin_mask.astype('uint8'), 'L')  # For binary masks ('L' = grayscale)

            # Apply transforms
            if color_transform:
                img_transformed = color_transform(transform(img_pil))
            else:
                img_transformed = transform(img_pil)
            bin_mask_transformed = transform(bin_mask_pil)

            # Convert back to numpy arrays
            img_transformed = np.array(img_transformed)
            bin_mask_transformed = np.array(bin_mask_transformed)

            save_mask(img_transformed, new_img_path, "png")
            save_mask(bin_mask_transformed, new_bin_path, "png")
            augmented_files.append((new_img_path, img_transformed, bin_mask_transformed, color_mask, xml_path))

    return augmented_files


def calculate_padding(dimension, patch_size):
    if dimension % patch_size == 0:
        return 0
    return patch_size - (dimension % patch_size)


def create_patches(file_pairs, config, data_dir):
    patch_img_dir = data_dir / "prepared_images"
    patch_bin_dir = data_dir / "prepared_binary_mask"

    patch_img_dir.mkdir(parents=True, exist_ok=True)
    patch_bin_dir.mkdir(parents=True, exist_ok=True)

    patches_files = []
    for img_path, img, bin_mask, color_mask, xml_path in tqdm(file_pairs, desc="Creating patches"):
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        if not isinstance(bin_mask, np.ndarray):
            bin_mask = np.array(bin_mask)

        pad_h = calculate_padding(img.shape[0], config["size_x"])
        pad_w = calculate_padding(img.shape[1], config["size_y"])

        img = np.pad(
            img,
            pad_width=((0, pad_h), (0, pad_w), (0, 0)),
            mode='constant',
            constant_values=0
        )
        bin_mask = np.pad(
            bin_mask,
            pad_width=((0, pad_h), (0, pad_w)),
            mode='constant',
            constant_values=0
        )

        img_patches = patchify(
            img,
            (
                (config["size_x"], config["size_y"], 3)
                if len(img.shape) == 3
                else (config["size_x"], config["size_y"])
            ),
            step=config["step"],
        )
        bin_patches = patchify(
            bin_mask, (config["size_x"], config["size_y"]), step=config["step"]
        )

        base_name = Path(img_path).stem

        for i in range(img_patches.shape[0]):
            for j in range(img_patches.shape[1]):
                img_patch_path = patch_img_dir / f"{base_name}_patch_{i}_{j}.png"
                bin_patch_path = patch_bin_dir / f"{base_name}_patch_{i}_{j}.png"

                img_patch = img_patches[i, j].squeeze()
                bin_patch = bin_patches[i, j].squeeze()

                if np.sum(bin_patch) == 0:
                    continue

                if bin_patch.dtype == np.float32 or bin_patch.dtype == np.float64:
                    bin_patch = (bin_patch * 255).astype(np.uint8)
                elif bin_patch.dtype != np.uint8:
                    bin_patch = bin_patch.astype(np.uint8)

                if img_patch.dtype != np.uint8:
                    img_patch = img_patch.astype(np.uint8)

                Image.fromarray(img_patch).save(img_patch_path)
                Image.fromarray(bin_patch).save(bin_patch_path)

                patches_files.append(
                    (str(img_patch_path), img_patch, bin_patch, "", xml_path)
                )

    return patches_files


def shrink_vertices(vertices, shrink_factor=0.8):
    """
    Shrink polygon vertices towards their centroid.

    Args:
        vertices: Numpy array of vertex coordinates
        shrink_factor: Float between 0 and 1, how much to shrink (1 = no shrink, 0 = shrink to center)

    Returns:
        Shrunk vertices array
    """
    centroid = np.mean(vertices, axis=0)
    shrunk_vertices = centroid + shrink_factor * (vertices - centroid)
    return shrunk_vertices


def poly2mask(vertex_row_coords, vertex_col_coords, shape, min_area=80, circle_radius=5):
    """Create mask from polygon vertices (similar to MATLAB's poly2mask)."""
    fill_row_coords, fill_col_coords = polygon(
        vertex_row_coords, vertex_col_coords, shape
    )
    mask = np.zeros(shape, dtype=np.uint8)
    mask[fill_row_coords, fill_col_coords] = 1
    area = np.sum(mask)
    if area < min_area:
        # Calculate center of polygon
        center_row = int(np.mean(vertex_row_coords))
        center_col = int(np.mean(vertex_col_coords))

        # Add circle
        circle_row, circle_col = disk((center_row, center_col), circle_radius, shape=shape)
        mask[circle_row, circle_col] = 1
    return mask


def get_boarder(mask):
    dilated = binary_dilation(mask)
    border = dilated - mask
    return border


def he_to_binary_mask(im_file, xml_file, config):
    with Image.open(im_file) as img:
        ncol, nrow = img.size

    tree = ET.parse(xml_file)
    root = tree.getroot()
    regions = root.findall(".//Region")

    binary_mask = np.zeros((nrow, ncol), dtype=np.float32)
    boarder_mask = np.zeros((nrow, ncol), dtype=np.float32)
    color_mask = np.zeros((nrow, ncol, 3), dtype=np.float32)
    overlap_mask = np.zeros((nrow, ncol), dtype=np.float32)

    cell_areas = []
    cell_perimeters = []

    for zz, region in enumerate(regions, 1):
        vertices = []
        for vertex in region.findall(".//Vertex"):
            x = float(vertex.get("X"))
            y = float(vertex.get("Y"))
            vertices.append([x, y])
        vertices = np.array(vertices)

        if "shrink" in config and config["shrink"]["use"]:
            shrunk_vertices = shrink_vertices(vertices, config["shrink"]["factor"])
            smaller_x = shrunk_vertices[:, 0]
            smaller_y = shrunk_vertices[:, 1]
        else:
            smaller_x = vertices[:, 0]
            smaller_y = vertices[:, 1]

        polygon_mask = poly2mask(smaller_y, smaller_x, (nrow, ncol))
        boarder_mask += get_boarder(polygon_mask)
        cell_areas.append(np.sum(polygon_mask))

        overlap_mask[np.logical_and(binary_mask > 0, polygon_mask > 0)] = 1.0
        binary_mask = (
            binary_mask + polygon_mask
        )  #  * zz * (1 - np.minimum(1, binary_mask))

        perimeter = np.sum(np.sqrt(np.diff(smaller_x) ** 2 + np.diff(smaller_y) ** 2))
        cell_perimeters.append(perimeter)

        random_color = np.random.rand(3)
        for i in range(3):
            color_mask[:, :, i] = color_mask[:, :, i] + random_color[i] * polygon_mask

    filename = Path(im_file).stem
    cell_info = {
        "id": filename,
        "num_cells": len(regions),
        "mean_cell_area": np.mean(cell_areas),
        "std_cell_area": np.std(cell_areas),
        "mean_cell_perimeter": np.mean(cell_perimeters),
        "std_cell_perimeter": np.std(cell_perimeters),
        "total_cell_area": np.sum(cell_areas),
        "image_width": ncol,
        "image_height": nrow,
        "overlap_area": np.sum(overlap_mask),
    }

    return binary_mask, color_mask, overlap_mask, boarder_mask, cell_info


def show_mask(binary_mask, color_mask, save_path=None):
    """Show binary mask and color mask if save_path is None. If save_path is give save it."""
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    img = Image.open(color_mask)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(132)
    binary_mask = Image.open(binary_mask)
    plt.imshow(binary_mask)
    plt.title("Binary Mask\n(Unique value per region)")
    plt.axis("off")

    plt.subplot(133)
    plt.imshow(np.clip(color_mask, 0, 1))
    plt.title("Color Mask\n(Random color per region)")
    plt.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def parse_vertices(region):
    """Extract vertex coordinates from a region."""
    vertices = []
    for vertex in region.findall(".//Vertex"):
        x = float(vertex.get("X"))
        y = float(vertex.get("Y"))
        vertices.append((x, y))
    return vertices


def create_binary_mask(xml_content, image_shape):
    """Create a binary mask from XML annotations."""
    root = ET.fromstring(xml_content)
    mask = Image.new("L", image_shape, 0)
    draw = ImageDraw.Draw(mask)

    for region in root.findall(".//Region"):
        vertices = parse_vertices(region)
        if vertices:
            # Draw the polygon on the mask
            draw.polygon(vertices, fill=255)

    return np.array(mask)


def save_mask(mask, output_path, out_format="png"):
    """Save mask array as image file."""
    output_path = f"{output_path}.{out_format}"
    if out_format == "npy":
        np.save(output_path, mask)
    else:
        # For image formats, scale to 0-255 if float
        if mask.dtype == np.float32 or mask.dtype == np.float64:
            mask = (mask * 255).astype(np.uint8)
        Image.fromarray(mask).save(output_path)
    return output_path


def overlay_mask_on_image(image, mask, alpha=0.5):
    """Overlay binary mask on the original image."""
    mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    mask_rgb[mask > 0] = [255, 0, 0]

    if len(image.shape) == 2:
        image_rgb = np.stack([image] * 3, axis=-1)
    else:
        image_rgb = image

    blend = (1 - alpha) * image_rgb + alpha * mask_rgb
    return blend.astype(np.uint8)


def get_matching_files(xml_dir, tif_dir):
    """
    Find matching XML and TIF files based on their base names.
    Returns a list of tuples (xml_path, tif_path).
    """
    xml_files = set(f[:-4] for f in os.listdir(xml_dir) if f.endswith(".xml"))
    tif_files = set(f[:-4] for f in os.listdir(tif_dir) if f.endswith(".tif"))

    common_names = xml_files & tif_files

    return [
        (os.path.join(xml_dir, name + ".xml"), os.path.join(tif_dir, name + ".tif"))
        for name in common_names
    ]


def augment_data_after_masking(file_pairs, methods, data_dir, cell_info):
    AUGMENTATION_METHODS = {
        # name: function
        "augment_random": augment_random,
    }
    if not methods:
        return file_pairs

    for method_name in methods:
        augment_func = AUGMENTATION_METHODS.get(method_name)
        if augment_func is None:
            logging.warning(f"Unknown augmentation method: {method_name}")
            continue

        file_pairs = augment_func(file_pairs, methods[method_name], data_dir, cell_info)

    return file_pairs


def augment_data_before_masking(file_pairs, methods, data_dir):
    AUGMENTATION_METHODS = {
        # name: function
    }

    if not methods:
        return file_pairs

    for method_name in methods:
        augment_func = AUGMENTATION_METHODS.get(method_name)
        if augment_func is not None and augment_func["after"]:
            continue
        if augment_func is None:
            logging.warning(f"Unknown augmentation method: {method_name}")
            continue

        file_pairs = augment_func(file_pairs, methods[method_name], data_dir)

    return file_pairs


def _create_sub_dirs(base_dir):
    binary_mask_dir = base_dir / "processed/binary_masks"
    color_mask_dir = base_dir / "processed/color_masks"
    overlap_mask_dir = base_dir / "processed/overlap"
    boarder_mask_dir = base_dir / "processed/boarder"
    no_overlap_mask_dir = base_dir / "processed/no_overlap"
    os.makedirs(binary_mask_dir, exist_ok=True)
    os.makedirs(color_mask_dir, exist_ok=True)
    os.makedirs(overlap_mask_dir, exist_ok=True)
    os.makedirs(boarder_mask_dir, exist_ok=True)
    os.makedirs(no_overlap_mask_dir, exist_ok=True)
    return (
        binary_mask_dir,
        boarder_mask_dir,
        color_mask_dir,
        overlap_mask_dir,
        no_overlap_mask_dir,
    )


def main(config):
    """
    final images and binary masks are always in  <set-type>/prepared_binary_mask or <set-type>/prepared_images.
    Args:
        config: Configuration dictionary

    Returns:
        None
    """
    set_type = config["dataset_split"]
    out_format = config["data"]["out_format"]
    data_c = config["data"][set_type]

    base_dir = config["dir"] / set_type
    binary_dir, boarder_dir, color_dir, overlap_dir, no_overlap_dir = _create_sub_dirs(base_dir)

    # Create prepared_images directory
    prepared_images_dir = base_dir / "prepared_images"
    prepared_images_dir.mkdir(parents=True, exist_ok=True)
    cell_info = pd.DataFrame()

    if "take_preprocessed" in config["data"] and config["data"]["take_preprocessed"]:
        logger.info(f"Skipping preprocessing for {set_type}")
        prepared_images_dir = config["data_dir"] / set_type / "prepared_images"
        prepared_mask_dir = config["data_dir"] / set_type / "prepared_binary_masks"
        prepared_images = os.listdir(prepared_images_dir)
        prepared_masks = os.listdir(prepared_mask_dir)

        processed_files = []
        for img_path, mask_path in zip(prepared_images, prepared_masks):
            img = Image.open(prepared_images_dir / img_path)
            mask = Image.open(prepared_mask_dir / mask_path)
            processed_files.append((img_path, img, mask, None, None))
    else:
        file_pairs = get_matching_files(
            config["data_dir"] / set_type / "Annotations",
            config["data_dir"] / set_type / "Tissue Images",
        )

        if data_c["limit"]:
            file_pairs = file_pairs[: data_c["limit"]]

        if "shrink" in config["data"]:
            data_c.update({"shrink": config["data"]["shrink"]})

        file_pairs = augment_data_before_masking(
            file_pairs, data_c["augment_before"], base_dir
        )

        processed_files = []
        for xml_path, tif_path in tqdm(
            file_pairs, desc=f"Processing images for {set_type}"
        ):
            binary_mask, color_mask, overlap_mask, boarder_mask, stats = he_to_binary_mask(tif_path, xml_path, data_c)
            cell_info = pd.concat([cell_info, pd.DataFrame([stats]).set_index("id")])
            binary_mask_no_overlap = (binary_mask > 0).astype(np.float32) - overlap_mask
            img = np.array(Image.open(tif_path))

            base_name = Path(xml_path).stem
            save_mask(binary_mask, binary_dir / base_name, out_format)
            save_mask(color_mask, color_dir / base_name, out_format)
            save_mask(boarder_mask, boarder_dir / base_name, out_format)
            save_mask(overlap_mask, overlap_dir / base_name, out_format)
            save_mask(binary_mask_no_overlap, no_overlap_dir / base_name, out_format)

            processed_files.append((base_name, img, binary_mask, color_mask, xml_path))

    cell_info.to_csv(base_dir / "actual_cell_counts.csv", index=True)

    if "patch" in config["data"] and config["data"]["patch"]:
        patches_files = create_patches(processed_files, config["data"]["patch"], base_dir)
        file_pairs = augment_data_after_masking(patches_files, data_c["augment_after"], base_dir, cell_info)
    else:
        for xml_path, tif_path in tqdm(
            file_pairs, desc="Copying files to prepared directory"
        ):
            base_name = Path(tif_path).stem

            img = Image.open(tif_path)
            new_image_path = prepared_images_dir / f"{base_name}.{out_format}"
            img.save(new_image_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument(
        "-d", "--dataset_split", choices=["test", "train"], default="train"
    )
    args = parser.parse_args()
    exp = load_experiment_config(args.experiment)
    exp["dataset_split"] = args.dataset_split
    main(exp)
