import os
import shutil
from pathlib import Path
from argparse import ArgumentParser

# --- CONFIGURATION ---
parser = ArgumentParser(description="Gather images from a folder")
parser.add_argument("input", type=str, default="", help="Input folder")
parser.add_argument("output", type=str, default="", help="Output folder")
args = parser.parse_args()

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}  # Supported image formats

# --- SCRIPT ---
def gather_images(src_folder: str, dst_folder: str):
    src_folder = Path(src_folder)
    dst_folder = Path(dst_folder)
    dst_folder.mkdir(parents=True, exist_ok=True)

    count = 0
    for file_path in src_folder.rglob("*"):  # recursive search
        if file_path.suffix.lower() in image_extensions:
            # Avoid filename collisions by adding a numeric prefix if needed
            dst_file = dst_folder / file_path.name
            i = 1
            while dst_file.exists():
                dst_file = dst_folder / f"{i}_{file_path.name}"
                i += 1

            shutil.copy2(file_path, dst_file)
            count += 1

    print(f"Copied {count} images to {dst_folder}")

if __name__ == "__main__":
    input_folder = args.input
    output_folder = args.output
    gather_images(input_folder, output_folder)
