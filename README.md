# kex
kaka > kex

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download EMNIST data

Data is downloaded automatically on first run. You can also pre-download a specific split manually:

```bash
python download_emnist.py --split balanced
```

This saves `.npy` arrays to `data/<split>/`. The full EMNIST zip (~500 MB) is downloaded once and reused for all splits.

---

## Training

```bash
python train.py [--split SPLIT] [--epochs N] [--batch-size N] [--lr LR]
```

**Examples:**

```bash
python train.py                                  # defaults: balanced, 10 epochs
python train.py --split digits --epochs 20
python train.py --split letters --batch-size 256 --lr 1e-3
```

Checkpoints are saved to `checkpoints/<split>/best.keras`, the final model to `saved_models/<split>/model.keras`, and TensorBoard logs to `logs/<split>/`.

---

## EMNIST splits

EMNIST extends MNIST to handwritten letters and digits. Choose a split based on your task:

| Split | Classes | Description |
|-------|---------|-------------|
| `balanced` | 47 | Digits + upper/lowercase letters with ambiguous pairs merged (e.g. C/c, O/o). Each class has equal sample count. **Good default.** |
| `byclass` | 62 | All digits + uppercase + lowercase, unbalanced. Largest split. |
| `bymerge` | 47 | Same 47 classes as balanced but without per-class balancing. |
| `letters` | 26 | Lowercase a–z only (case-insensitive). Simplest letter task. |
| `digits` | 10 | Digits 0–9 only. Drop-in replacement for MNIST. |
| `mnist` | 10 | MNIST-compatible subset, same images as original MNIST. |

**Which to use:**
- Start with `balanced` — manageable class count, even training distribution.
- Use `digits` if you only care about numbers (fastest training).
- Use `letters` for a pure letter recognition task (26 classes).
- Use `byclass` for the hardest, most general task (62 classes, large dataset).
