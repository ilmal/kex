"""
Download and preprocess the EMNIST dataset.

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
import os
import struct
import gzip
import urllib.request
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# EMNIST is distributed as gzipped IDX files hosted by NIST (via BIOMETRICS).
# The official mirror used here is from the NIST website.
# ---------------------------------------------------------------------------
BASE_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
GDRIVE_MIRROR = "https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip"

# Convenience: pre-built per-split URLs (Cohen et al. dataset page on Google Drive)
# Falls back to downloading the full zip above if these fail.
SPLIT_FILES = {
    # (split_name) -> (train_images, train_labels, test_images, test_labels)
    "balanced": (
        "emnist-balanced-train-images-idx3-ubyte.gz",
        "emnist-balanced-train-labels-idx1-ubyte.gz",
        "emnist-balanced-test-images-idx3-ubyte.gz",
        "emnist-balanced-test-labels-idx1-ubyte.gz",
    ),
    "byclass": (
        "emnist-byclass-train-images-idx3-ubyte.gz",
        "emnist-byclass-train-labels-idx1-ubyte.gz",
        "emnist-byclass-test-images-idx3-ubyte.gz",
        "emnist-byclass-test-labels-idx1-ubyte.gz",
    ),
    "bymerge": (
        "emnist-bymerge-train-images-idx3-ubyte.gz",
        "emnist-bymerge-train-labels-idx1-ubyte.gz",
        "emnist-bymerge-test-images-idx3-ubyte.gz",
        "emnist-bymerge-test-labels-idx1-ubyte.gz",
    ),
    "letters": (
        "emnist-letters-train-images-idx3-ubyte.gz",
        "emnist-letters-train-labels-idx1-ubyte.gz",
        "emnist-letters-test-images-idx3-ubyte.gz",
        "emnist-letters-test-labels-idx1-ubyte.gz",
    ),
    "digits": (
        "emnist-digits-train-images-idx3-ubyte.gz",
        "emnist-digits-train-labels-idx1-ubyte.gz",
        "emnist-digits-test-images-idx3-ubyte.gz",
        "emnist-digits-test-labels-idx1-ubyte.gz",
    ),
    "mnist": (
        "emnist-mnist-train-images-idx3-ubyte.gz",
        "emnist-mnist-train-labels-idx1-ubyte.gz",
        "emnist-mnist-test-images-idx3-ubyte.gz",
        "emnist-mnist-test-labels-idx1-ubyte.gz",
    ),
}

NUM_CLASSES = {
    "balanced": 47,
    "byclass": 62,
    "bymerge": 47,
    "letters": 26,
    "digits": 10,
    "mnist": 10,
}


def _download_zip(dest_dir: Path) -> Path:
    """Download the full EMNIST gzip bundle and extract it."""
    import zipfile

    zip_path = dest_dir / "gzip.zip"
    if not zip_path.exists():
        print(f"Downloading EMNIST bundle to {zip_path} ...")
        try:
            urllib.request.urlretrieve(BASE_URL, zip_path, reporthook=_progress)
        except Exception:
            print("Primary mirror failed, trying secondary...")
            urllib.request.urlretrieve(GDRIVE_MIRROR, zip_path, reporthook=_progress)
        print()

    extract_dir = dest_dir / "gzip"
    if not extract_dir.exists():
        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
    return extract_dir


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        print(f"\r  {pct:5.1f}%  ({downloaded // 1024 // 1024} MB / {total_size // 1024 // 1024} MB)", end="")


def _read_idx_images(gz_path: Path) -> np.ndarray:
    with gzip.open(gz_path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 0x0803, f"Bad magic number {magic:#010x} in {gz_path}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows, cols)


def _read_idx_labels(gz_path: Path) -> np.ndarray:
    with gzip.open(gz_path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 0x0801, f"Bad magic number {magic:#010x} in {gz_path}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def _fix_rotation(images: np.ndarray) -> np.ndarray:
    """EMNIST images are stored transposed; fix to upright orientation."""
    return np.transpose(images, (0, 2, 1))


def download_and_save(split: str = "balanced", data_root: str = "data") -> dict:
    """
    Download (if needed) and save EMNIST split as .npy arrays.

    Returns a dict with keys: x_train, y_train, x_test, y_test.
    """
    if split not in SPLIT_FILES:
        raise ValueError(f"Unknown split '{split}'. Choose from: {list(SPLIT_FILES)}")

    data_root = Path(data_root)
    raw_dir = data_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    gz_dir = _download_zip(raw_dir)

    train_img_f, train_lbl_f, test_img_f, test_lbl_f = SPLIT_FILES[split]

    print(f"Loading {split} split...")
    x_train = _fix_rotation(_read_idx_images(gz_dir / train_img_f))
    y_train = _read_idx_labels(gz_dir / train_lbl_f)
    x_test = _fix_rotation(_read_idx_images(gz_dir / test_img_f))
    y_test = _read_idx_labels(gz_dir / test_lbl_f)

    # Normalise to [0, 1] float32
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    out_dir = data_root / split
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
        choices=list(SPLIT_FILES),
        help="EMNIST split to download (default: balanced)",
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Root directory for dataset storage (default: data/)",
    )
    args = parser.parse_args()
    download_and_save(split=args.split, data_root=args.data_root)
