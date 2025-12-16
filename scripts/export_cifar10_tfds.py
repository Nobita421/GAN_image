"""Export CIFAR-10 from TensorFlow Datasets (TFDS) into a folder structure.

Exports:
  <output_dir>/train/*.png
  <output_dir>/test/*.png

This is useful when you want to keep using the existing "load images from folder"
training pipeline, but source the images from TFDS.
"""

import argparse
import os
from typing import Optional

import yaml


def _load_config(path: str) -> dict:
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _export_split(ds, out_dir: str, max_images: Optional[int]) -> int:
    from PIL import Image

    _ensure_dir(out_dir)

    written = 0
    for idx, (image, _label) in enumerate(ds):
        if max_images is not None and idx >= max_images:
            break

        # image is uint8 [H,W,3]
        img = Image.fromarray(image)
        out_path = os.path.join(out_dir, f"{idx:06d}.png")
        img.save(out_path)
        written += 1

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CIFAR-10 via TFDS into PNG folders")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (used for export_dir default)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory root (default: export_dir from config.yaml, else data/raw)",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Max images to export PER split (train and test). Useful for quick runs.",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    output_dir = args.output_dir or cfg.get("export_dir") or "data/raw"

    try:
        import tensorflow_datasets as tfds
    except Exception as e:
        raise RuntimeError(
            "tensorflow_datasets is required. Install it with: pip install tensorflow-datasets"
        ) from e

    print("Loading CIFAR-10 via TFDS...")
    train_ds = tfds.load("cifar10", split="train", as_supervised=True)
    test_ds = tfds.load("cifar10", split="test", as_supervised=True)

    # Convert to NumPy for simple PIL saving
    train_np = tfds.as_numpy(train_ds)
    test_np = tfds.as_numpy(test_ds)

    train_out = os.path.join(output_dir, "train")
    test_out = os.path.join(output_dir, "test")

    print(f"Exporting train split to: {train_out}")
    n_train = _export_split(train_np, train_out, args.max_images)

    print(f"Exporting test split to: {test_out}")
    n_test = _export_split(test_np, test_out, args.max_images)

    print("Done.")
    print(f"Wrote {n_train} train images and {n_test} test images into {output_dir}")


if __name__ == "__main__":
    main()
