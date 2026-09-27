"""
Synthetic dataset generator utility for visual product search demonstrations.
Creates catalog categories with synthetic pattern-based product images.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import cv2
from utils.logger import get_logger

logger = get_logger(__name__)


def create_synthetic_catalog(
    output_dir: Path,
    num_samples_per_category: int = 15,
    image_size: Tuple[int, int] = (224, 224)
) -> Dict[str, List[Path]]:
    """Generates synthetic product catalog images across multiple product categories.

    Categories: 'shoes', 'shirts', 'watches', 'bags', 'sunglasses'

    Args:
        output_dir: Target directory path to store generated image folders.
        num_samples_per_category: Number of synthetic images to create per category.
        image_size: Target dimensions (height, width) of output images.

    Returns:
        Dict[str, List[Path]]: Mapping of category names to generated image file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = {
        "shoes": (0, 100, 255),      # Orange/Brown tones
        "shirts": (255, 100, 0),     # Blue tones
        "watches": (50, 200, 50),     # Green/Metallic tones
        "bags": (200, 50, 200),      # Purple/Magenta tones
        "sunglasses": (50, 50, 50)   # Dark/Grey tones
    }

    generated_files: Dict[str, List[Path]] = {cat: [] for cat in categories}

    for cat_name, base_color in categories.items():
        cat_dir = output_dir / cat_name
        cat_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(1, num_samples_per_category + 1):
            img = np.full((image_size[0], image_size[1], 3), 245, dtype=np.uint8)

            # Draw distinct category shapes to create visual similarity clusters
            color_variation = (
                max(0, min(255, base_color[0] + np.random.randint(-30, 30))),
                max(0, min(255, base_color[1] + np.random.randint(-30, 30))),
                max(0, min(255, base_color[2] + np.random.randint(-30, 30)))
            )

            center = (image_size[1] // 2 + np.random.randint(-15, 15), image_size[0] // 2 + np.random.randint(-15, 15))

            if cat_name == "shoes":
                # Draw shoe-like ellipse
                cv2.ellipse(img, center, (70, 35), 20, 0, 360, color_variation, -1)
                cv2.rectangle(img, (center[0]-40, center[1]+10), (center[0]+50, center[1]+30), (100, 100, 100), -1)
            elif cat_name == "shirts":
                # Draw shirt rectangle with collar lines
                cv2.rectangle(img, (center[0]-50, center[1]-60), (center[0]+50, center[1]+60), color_variation, -1)
                cv2.line(img, (center[0]-25, center[1]-60), (center[0], center[1]-30), (255, 255, 255), 4)
                cv2.line(img, (center[0]+25, center[1]-60), (center[0], center[1]-30), (255, 255, 255), 4)
            elif cat_name == "watches":
                # Draw circular watch dial with strap
                cv2.rectangle(img, (center[0]-15, center[1]-80), (center[0]+15, center[1]+80), (60, 60, 60), -1)
                cv2.circle(img, center, 45, color_variation, -1)
                cv2.circle(img, center, 35, (255, 255, 255), -1)
            elif cat_name == "bags":
                # Draw bag body with handle
                cv2.rectangle(img, (center[0]-55, center[1]-20), (center[0]+55, center[1]+60), color_variation, -1)
                cv2.ellipse(img, (center[0], center[1]-20), (30, 25), 0, 180, 360, (50, 50, 50), 6)
            elif cat_name == "sunglasses":
                # Draw sunglasses lenses and frame bridge
                cv2.circle(img, (center[0]-30, center[1]), 25, color_variation, -1)
                cv2.circle(img, (center[0]+30, center[1]), 25, color_variation, -1)
                cv2.line(img, (center[0]-30, center[1]), (center[0]+30, center[1]), (20, 20, 20), 4)

            # Add subtle texture noise
            noise = np.random.randint(-15, 15, (image_size[0], image_size[1], 3), dtype=np.int16)
            img_noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            file_path = cat_dir / f"{cat_name}_{idx:03d}.jpg"
            cv2.imwrite(str(file_path), cv2.cvtColor(img_noisy, cv2.COLOR_RGB2BGR))
            generated_files[cat_name].append(file_path)

    logger.info(f"Generated synthetic product catalog at {output_dir} with {sum(len(v) for v in generated_files.values())} images.")
    return generated_files


if __name__ == "__main__":
    from config.config import config
    create_synthetic_catalog(config.paths.raw_data_dir)
