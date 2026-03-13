"""
Train a simple dense (MLP) neural network on EMNIST.

Usage:
  python train.py                          # defaults: balanced split, 10 epochs
  python train.py --split digits --epochs 20
  python train.py --split letters --batch-size 256 --lr 1e-3
"""

import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import download_emnist

# ---------------------------------------------------------------------------
# Number of output classes per split
# ---------------------------------------------------------------------------
NUM_CLASSES = download_emnist.NUM_CLASSES


def build_model(input_shape: tuple, num_classes: int) -> keras.Model:
    """Simple dense MLP: Flatten -> Dense x3 with BatchNorm + Dropout -> Softmax."""
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="emnist_mlp",
    )
    return model


def main(args):
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"Loading EMNIST '{args.split}' split...")
    data = download_emnist.load(split=args.split, data_root=args.data_root)

    x_train, y_train = data["x_train"], data["y_train"]
    x_test, y_test = data["x_test"], data["y_test"]

    # Add channel dim for consistency (28, 28) -> (28, 28, 1)
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    num_classes = NUM_CLASSES[args.split]

    print(f"Train: {x_train.shape}, Test: {x_test.shape}, Classes: {num_classes}")

    # ------------------------------------------------------------------
    # 2. Build model
    # ------------------------------------------------------------------
    model = build_model(input_shape=x_train.shape[1:], num_classes=num_classes)
    model.summary()

    optimizer = keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # ------------------------------------------------------------------
    # 3. Callbacks
    # ------------------------------------------------------------------
    checkpoint_dir = os.path.join("checkpoints", args.split)
    os.makedirs(checkpoint_dir, exist_ok=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "best.keras"),
            save_best_only=True,
            monitor="val_accuracy",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join("logs", args.split),
            histogram_freq=1,
        ),
    ]

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    history = model.fit(
        x_train,
        y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    # ------------------------------------------------------------------
    # 5. Evaluate on test set
    # ------------------------------------------------------------------
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy : {test_acc:.4f}")
    print(f"Test loss     : {test_loss:.4f}")

    # ------------------------------------------------------------------
    # 6. Save final model
    # ------------------------------------------------------------------
    saved_dir = os.path.join("saved_models", args.split)
    os.makedirs(saved_dir, exist_ok=True)
    model.save(os.path.join(saved_dir, "model.keras"))
    print(f"Model saved to {saved_dir}/model.keras")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP on EMNIST")
    parser.add_argument(
        "--split",
        default="balanced",
        choices=list(NUM_CLASSES),
        help="EMNIST split (default: balanced)",
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Root directory for dataset (default: data/)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Mini-batch size (default: 128)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate (default: 1e-3)",
    )
    args = parser.parse_args()
    main(args)
