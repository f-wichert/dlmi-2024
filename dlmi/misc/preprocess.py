from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from skimage.draw import polygon
from dlmi.shared.config import load_config
import os
from tqdm import tqdm
import logging


logger = logging.getLogger(__name__)

AUGMENTATION_METHODS = {
    # name: function
}


def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    """Create mask from polygon vertices (similar to MATLAB's poly2mask)."""
    fill_row_coords, fill_col_coords = polygon(vertex_row_coords, vertex_col_coords, shape)
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
    regions = root.findall('.//Region')

    binary_mask = np.zeros((nrow, ncol), dtype=np.float32)
    color_mask = np.zeros((nrow, ncol, 3), dtype=np.float32)

    for zz, region in enumerate(regions, 1):
        # Extract vertices
        vertices = []
        for vertex in region.findall('.//Vertex'):
            x = float(vertex.get('X'))
            y = float(vertex.get('Y'))
            vertices.append([x, y])
        vertices = np.array(vertices)

        # Split into x and y coordinates
        smaller_x = vertices[:, 0]
        smaller_y = vertices[:, 1]

        polygon_mask = poly2mask(smaller_y, smaller_x, (nrow, ncol))

        binary_mask = binary_mask + zz * (1 - np.minimum(1, binary_mask)) * polygon_mask

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
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(132)
    binary_mask = Image.open(binary_mask)
    plt.imshow(binary_mask)
    plt.title('Binary Mask\n(Unique value per region)')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(np.clip(color_mask, 0, 1))
    plt.title('Color Mask\n(Random color per region)')
    plt.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def parse_vertices(region):
    """Extract vertex coordinates from a region."""
    vertices = []
    for vertex in region.findall('.//Vertex'):
        x = float(vertex.get('X'))
        y = float(vertex.get('Y'))
        vertices.append((x, y))
    return vertices


def create_binary_mask(xml_content, image_shape):
    """Create a binary mask from XML annotations."""
    # Parse XML
    root = ET.fromstring(xml_content)

    # Create an empty mask
    mask = Image.new('L', image_shape, 0)
    draw = ImageDraw.Draw(mask)

    # Process each region
    for region in root.findall('.//Region'):
        vertices = parse_vertices(region)
        if vertices:
            # Draw the polygon on the mask
            draw.polygon(vertices, fill=255)

    return np.array(mask)


def save_mask(mask, output_path, format='png'):
    """Save mask array as image file."""
    if format == 'npy':
        np.save(output_path, mask)
    else:
        # For image formats, scale to 0-255 if float
        if mask.dtype == np.float32 or mask.dtype == np.float64:
            mask = (mask * 255).astype(np.uint8)
        Image.fromarray(mask).save(output_path)


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
    xml_files = set(f[:-4] for f in os.listdir(xml_dir) if f.endswith('.xml'))
    tif_files = set(f[:-4] for f in os.listdir(tif_dir) if f.endswith('.tif'))

    common_names = xml_files & tif_files

    return [(
        os.path.join(xml_dir, name + '.xml'),
        os.path.join(tif_dir, name + '.tif')
    ) for name in common_names]


def augment_data(file_pairs, methods):
    augmented_pairs = file_pairs.copy()

    for method_name in methods:
        try:
            augment_func = AUGMENTATION_METHODS.get(method_name)
            if augment_func is None:
                logging.warning(f"Unknown augmentation method: {method_name}")
                continue

            new_pairs = []
            for xml_path, image_path in file_pairs:
                try:
                    aug_image_path, aug_xml_path = augment_func(
                        Path(image_path),
                        Path(xml_path)
                    )
                    new_pairs.append((str(aug_xml_path), str(aug_image_path)))
                except Exception as e:
                    logger.error(f"Failed to augment {image_path} with {method_name}: {str(e)}")
                    continue

            augmented_pairs.extend(new_pairs)

        except Exception as e:
            logging.error(f"Error applying augmentation method {method_name}: {str(e)}")
            continue

    return augmented_pairs


def main():
    config = load_config()
    for set_type in config['data']['prepare_for']:
        ic = config['data'][set_type]
        base_dir = Path(config['data']['data_dir']) / ic['dir']
        binary_mask_dir = base_dir / ic['binary_masks']
        color_mask_dir = base_dir / ic['color_masks']

        # Create directories
        os.makedirs(binary_mask_dir, exist_ok=True)
        os.makedirs(color_mask_dir, exist_ok=True)

        # Get matching files
        file_pairs = get_matching_files(
            base_dir / ic['annotations_xml_dir'],
            base_dir / ic['images_tif_dir']
        )
        if len(file_pairs) == 0:
            raise RuntimeError()
        if ic.get('limit', None):
            file_pairs = file_pairs[:ic['limit']]

        file_pairs = augment_data(file_pairs, ic['augment'])

        # Process each pair
        for xml_path, tif_path in tqdm(file_pairs, desc=f"Processing images for {set_type}"):
            binary_mask, color_mask = he_to_binary_mask(tif_path, xml_path)

            base_name = Path(xml_path).stem
            binary_out = binary_mask_dir / f"{base_name}.{config['data']['out_format']}"
            color_out = color_mask_dir / f"{base_name}.{config['data']['out_format']}"

            save_mask(binary_mask, binary_out, config['data']['out_format'])
            save_mask(color_mask, color_out, config['data']['out_format'])


if __name__ == '__main__':
    main()
