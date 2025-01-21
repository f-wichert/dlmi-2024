import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from patchify import patchify
from PIL import Image, ImageDraw
from skimage.draw import polygon
from tqdm import tqdm

from dlmi.shared.config import load_config

logger = logging.getLogger(__name__)


def create_patches(file_pairs, config, data_dir):
    patch_img_dir = data_dir / Path(config["image_dir"])
    patch_bin_dir = data_dir / Path(config["binary_dir"])

    patch_img_dir.mkdir(parents=True, exist_ok=True)
    patch_bin_dir.mkdir(parents=True, exist_ok=True)

    augmented_files = []
    for img_path, bin_mask, color_mask, xml_path in file_pairs:
        img = np.array(Image.open(img_path))

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

        # save patches
        for i in range(img_patches.shape[0]):
            for j in range(img_patches.shape[1]):
                img_patch_filename = f"{base_name}_patch_img_{i}_{j}.png"
                bin_patch_filename = f"{base_name}_patch_bin_{i}_{j}.png"

                img_patch_path = patch_img_dir / img_patch_filename
                bin_patch_path = patch_bin_dir / bin_patch_filename

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

                augmented_files.append(
                    (str(img_patch_path), str(bin_patch_path), "", xml_path)
                )

    return augmented_files


def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    """Create mask from polygon vertices (similar to MATLAB's poly2mask)."""
    fill_row_coords, fill_col_coords = polygon(
        vertex_row_coords, vertex_col_coords, shape
    )
    mask = np.zeros(shape, dtype=np.uint8)
    mask[fill_row_coords, fill_col_coords] = 1
    return mask


def he_to_binary_mask(im_file, xml_file):
    """
    Convert H&E image and XML annotations to binary and color masks.
    Replicates MATLAB function functionality.
    Based on: https://monuseg.grand-challenge.org/Data/
        -> https://drive.google.com/file/d/1YDtIiLZX0lQzZp_JbqneHXHvRo45ZWGX/view?usp=sharing

    Args:
        filename: Base filename without extension

    Returns:
        binary_mask: Mask where each region has a unique integer value
        color_mask: RGB mask where each region has a random color
    """
    with Image.open(im_file) as img:
        ncol, nrow = img.size

    tree = ET.parse(xml_file)
    root = tree.getroot()
    regions = root.findall(".//Region")

    binary_mask = np.zeros((nrow, ncol), dtype=np.float32)
    color_mask = np.zeros((nrow, ncol, 3), dtype=np.float32)

    for zz, region in enumerate(regions, 1):
        # extract vertices
        vertices = []
        for vertex in region.findall(".//Vertex"):
            x = float(vertex.get("X"))
            y = float(vertex.get("Y"))
            vertices.append([x, y])
        vertices = np.array(vertices)

        # split into x and y coordinates
        smaller_x = vertices[:, 0]
        smaller_y = vertices[:, 1]

        polygon_mask = poly2mask(smaller_y, smaller_x, (nrow, ncol))

        binary_mask = (
            binary_mask + polygon_mask
        )  #  * zz * (1 - np.minimum(1, binary_mask))

        random_color = np.random.rand(3)
        for i in range(3):
            color_mask[:, :, i] = color_mask[:, :, i] + random_color[i] * polygon_mask

    return binary_mask, color_mask


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
    # Parse XML
    root = ET.fromstring(xml_content)

    # Create an empty mask
    mask = Image.new("L", image_shape, 0)
    draw = ImageDraw.Draw(mask)

    # Process each region
    for region in root.findall(".//Region"):
        vertices = parse_vertices(region)
        if vertices:
            # Draw the polygon on the mask
            draw.polygon(vertices, fill=255)

    return np.array(mask)


def save_mask(mask, output_path, format="png"):
    """Save mask array as image file."""
    if format == "npy":
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


def augment_data_after_masking(file_pairs, methods, data_dir):
    AUGMENTATION_METHODS = {
        # name: function
        "patch": create_patches,
    }
    if not methods:
        return file_pairs

    for method_name in methods:
        augment_func = AUGMENTATION_METHODS.get(method_name)
        if augment_func is None:
            logging.warning(f"Unknown augmentation method: {method_name}")
            continue

        file_pairs = augment_func(file_pairs, methods[method_name], data_dir)

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


def main():
    config = load_config()
    for set_type in config["data"]["prepare_for"]:
        ic = config["data"][set_type]
        base_dir = Path(config["data"]["data_dir"]) / ic["dir"]
        binary_mask_dir = base_dir / ic["binary_masks"]
        color_mask_dir = base_dir / ic["color_masks"]

        # create directories
        os.makedirs(binary_mask_dir, exist_ok=True)
        os.makedirs(color_mask_dir, exist_ok=True)

        # get matching files
        file_pairs = get_matching_files(
            base_dir / ic["annotations_xml_dir"], base_dir / ic["images_tif_dir"]
        )
        if len(file_pairs) == 0:
            raise RuntimeError()
        if ic.get("limit", None):
            file_pairs = file_pairs[: ic["limit"]]

        file_pairs = augment_data_before_masking(
            file_pairs, ic["augment_before"], base_dir
        )

        new_files = []

        # process each pair
        for xml_path, tif_path in tqdm(
            file_pairs, desc=f"Processing images for {set_type}"
        ):
            binary_mask, color_mask = he_to_binary_mask(tif_path, xml_path)
            new_files.append((tif_path, binary_mask, color_mask, xml_path))
            base_name = Path(xml_path).stem
            binary_out = binary_mask_dir / f"{base_name}.{config['data']['out_format']}"
            color_out = color_mask_dir / f"{base_name}.{config['data']['out_format']}"

            save_mask(binary_mask, binary_out, config["data"]["out_format"])
            save_mask(color_mask, color_out, config["data"]["out_format"])

        augment_data_after_masking(new_files, ic["augment_after"], base_dir)


if __name__ == "__main__":
    main()
