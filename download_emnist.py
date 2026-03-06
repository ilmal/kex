"""
Download and preprocess the EMNIST dataset via tensorflow-datasets.

EMNIST splits available:
  - byclass   : 62 classes (digits + upper + lower)
  - bymerge   : 47 classes (merged similar letters)
  - balanced  : 47 classes, balanced per class        <-- default
  - letters   : 26 classes (a-z, case-insensitive)
  - digits    : 10 classes (0-9)
  - mnist     : 10 classes (same as original MNIST)

Saves numpy arrays to data/<split>/:
  x_train.npy, y_train.npy, x_test.npy, y_test.npy
"""

import argparse
from pathlib import Path

import numpy as np

NUM_CLASSES = {
    "balanced": 47,
    "byclass": 62,
    "bymerge": 47,
    "letters": 26,
    "digits": 10,
    "mnist": 10,
}


def _load_from_tfds(split: str, data_root: str) -> tuple:
    """Use tensorflow-datasets to download and return numpy arrays."""
    import tensorflow_datasets as tfds

    tfds_name = f"emnist/{split}"
    tfds_dir = str(Path(data_root) / "tfds")
    print(f"Downloading EMNIST '{split}' via tensorflow-datasets...")

    # batch_size=-1 loads the full split into memory as tensors
    train_data = tfds.load(tfds_name, split="train", batch_size=-1, data_dir=tfds_dir)
    test_data = tfds.load(tfds_name, split="test", batch_size=-1, data_dir=tfds_dir)

    train_np = tfds.as_numpy(train_data)
    test_np = tfds.as_numpy(test_data)

    # tfds images are (N, 28, 28, 1) uint8 and already correctly oriented
    x_train = train_np["image"].squeeze(-1).astype(np.float32) / 255.0
    y_train = train_np["label"].astype(np.uint8)
    x_test = test_np["image"].squeeze(-1).astype(np.float32) / 255.0
    y_test = test_np["label"].astype(np.uint8)

    return x_train, y_train, x_test, y_test


def download_and_save(split: str = "balanced", data_root: str = "data") -> dict:
    """
    Download (if needed) and save EMNIST split as .npy arrays.

    Returns a dict with keys: x_train, y_train, x_test, y_test.
    """
    if split not in NUM_CLASSES:
        raise ValueError(f"Unknown split '{split}'. Choose from: {list(NUM_CLASSES)}")

    x_train, y_train, x_test, y_test = _load_from_tfds(split, data_root)

    out_dir = Path(data_root) / split
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "x_train.npy", x_train)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "x_test.npy", x_test)
    np.save(out_dir / "y_test.npy", y_test)

    print(f"Saved to {out_dir}/")
    print(f"  x_train: {x_train.shape}  y_train: {y_train.shape}")
    print(f"  x_test:  {x_test.shape}   y_test:  {y_test.shape}")
    print(f"  Classes: {NUM_CLASSES[split]}")

    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}


def load(split: str = "balanced", data_root: str = "data") -> dict:
    """Load pre-saved .npy arrays; download first if missing."""
    out_dir = Path(data_root) / split
    files = ["x_train.npy", "y_train.npy", "x_test.npy", "y_test.npy"]
    if not all((out_dir / f).exists() for f in files):
        return download_and_save(split, data_root)
    return {
        "x_train": np.load(out_dir / "x_train.npy"),
        "y_train": np.load(out_dir / "y_train.npy"),
        "x_test": np.load(out_dir / "x_test.npy"),
        "y_test": np.load(out_dir / "y_test.npy"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download EMNIST dataset")
    parser.add_argument(
        "--split",
        default="balanced",
        choices=list(NUM_CLASSES),
        help="EMNIST split to download (default: balanced)",
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Root directory for dataset storage (default: data/)",
    )
    args = parser.parse_args()
    download_and_save(split=args.split, data_root=args.data_root)
