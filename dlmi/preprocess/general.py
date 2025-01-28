import logging
import os
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from skimage.draw import polygon
from skimage.morphology import binary_dilation
from tqdm import tqdm

from dlmi.utils.utils import load_experiment_config

logger = logging.getLogger(__name__)


def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    """Create mask from polygon vertices (similar to MATLAB's poly2mask)."""
    fill_row_coords, fill_col_coords = polygon(
        vertex_row_coords, vertex_col_coords, shape
    )
    mask = np.zeros(shape, dtype=np.uint8)
    mask[fill_row_coords, fill_col_coords] = 1
    return mask


def get_boarder(mask):
    dilated = binary_dilation(mask)
    border = dilated - mask
    return border


def he_to_binary_mask(im_file, xml_file, cell_info):
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

        smaller_x = vertices[:, 0]
        smaller_y = vertices[:, 1]

        polygon_mask = poly2mask(smaller_y, smaller_x, (nrow, ncol))
        boarder_mask += get_boarder(polygon_mask)
        cell_areas.append(np.sum(polygon_mask))

        overlap_mask[np.logical_and(binary_mask > 0, polygon_mask > 0)] = 1.0
        binary_mask = binary_mask + polygon_mask    #  * zz * (1 - np.minimum(1, binary_mask))

        perimeter = np.sum(np.sqrt(np.diff(smaller_x) ** 2 + np.diff(smaller_y) ** 2))
        cell_perimeters.append(perimeter)

        random_color = np.random.rand(3)
        for i in range(3):
            color_mask[:, :, i] = color_mask[:, :, i] + random_color[i] * polygon_mask

    filename = Path(im_file).stem
    cell_info[str(filename)] = {
       'num_cells': len(regions),
       'mean_cell_area': np.mean(cell_areas),
       'std_cell_area': np.std(cell_areas),
       'mean_cell_perimeter': np.mean(cell_perimeters),
       'std_cell_perimeter': np.std(cell_perimeters),
       'cell_density': len(regions) / (nrow * ncol),
       'total_cell_area': np.sum(cell_areas),
       'image_width': ncol,
       'image_height': nrow
    }

    return binary_mask, color_mask, overlap_mask, boarder_mask


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


def _create_sub_dirs(base_dir):
    binary_mask_dir = base_dir / "processed/binary_masks"
    color_mask_dir = base_dir / "processed/color_masks"
    overlap_mask_dir = base_dir / "processed/overlap"
    boarder_mask_dir = base_dir / "processed/boarder"
    cell_count_dir = base_dir / "processed/cell_counts"
    no_overlap_mask_dir = base_dir / "processed/no_overlap"
    os.makedirs(binary_mask_dir, exist_ok=True)
    os.makedirs(color_mask_dir, exist_ok=True)
    os.makedirs(cell_count_dir, exist_ok=True)
    os.makedirs(overlap_mask_dir, exist_ok=True)
    os.makedirs(boarder_mask_dir, exist_ok=True)
    return binary_mask_dir, boarder_mask_dir, cell_count_dir, color_mask_dir, overlap_mask_dir, no_overlap_mask_dir


def main(config):
    """
    final images and binary masks are always in  <set-type>/prepared_binary_mask or <set-type>/prepared_images.
    Args:
        config: Configuration dictionary

    Returns:
        None
    """
    set_type = config["dataset_split"]
    out_format = config['data']['out_format']
    data_c = config["data"][set_type]

    base_dir = config["dir"] / set_type
    binary_dir, boarder_dir, cell_count_dir, color_dir, overlap_dir, no_overlap_dir = _create_sub_dirs(base_dir)

    # Create prepared_images directory
    prepared_images_dir = base_dir / 'prepared_images'
    prepared_images_dir.mkdir(parents=True, exist_ok=True)

    cell_info = pd.DataFrame()

    file_pairs = get_matching_files(
        config["data_dir"] / set_type / "Annotations",
        config["data_dir"] / set_type / "Tissue Images"
    )
    if data_c["limit"]:
        file_pairs = file_pairs[: data_c["limit"]]

    processed_files = []

    for xml_path, tif_path in tqdm(file_pairs, desc=f"Processing images for {set_type}"):
        binary_mask, color_mask, overlap_mask, boarder_mask = he_to_binary_mask(tif_path, xml_path, cell_info)
        binary_mask_no_overlap = (binary_mask > 0).astype(np.float32) - overlap_mask

        base_name = Path(xml_path).stem

        save_mask(binary_mask, binary_dir / f"{base_name}", out_format)
        save_mask(color_mask, color_dir / f"{base_name}", out_format)
        save_mask(boarder_mask, boarder_dir / f"{base_name}-boarder", out_format)
        save_mask(overlap_mask, overlap_dir / f"{base_name}", out_format)
        save_mask(binary_mask_no_overlap, binary_dir / f"{base_name}-no-overlap", out_format)

        processed_files.append((tif_path, binary_mask, color_mask, xml_path))

    cell_info.to_csv(cell_count_dir / "cell_counts.csv")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument("-d", "--dataset_split", choices=["test", "train"], default="train")
    args = parser.parse_args()
    exp = load_experiment_config(args.experiment)
    exp["dataset_split"] = args.dataset_split
    main(exp)
