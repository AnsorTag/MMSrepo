# Image Classification Using Deep Neural Networks

This project implements and compares deep learning models for image classification using the Flowers102 dataset using PyTorch.

---

## Models Used

- **ResNet50** — higher accuracy
- **MobileNetV2** — more efficient and lightweight

---

## Features

- Transfer learning with pretrained CNNs
- Comparison of model performance
- Experiments with:
  - image size
  - dataset size
- Confusion matrix visualization

---

## Project Structure

```

src/
train.py
evaluate.py
dataset.py
model.py

data/        # dataset (NOT included in repo)
outputs/     # trained models (auto-created)
results/     # metrics and confusion matrices (auto-created)

```

---

## Setup (Recommended: Virtual Environment)

### 1. Create virtual environment

```bash
python -m venv .venv
```

### 2. Activate it

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset

This project uses the **Flowers102 dataset**.

⚠️ The dataset is NOT included due to size.

You must:

* download the dataset manually
* place it inside the `data/` folder

Example:

```
data/
  train/
  val/
  test/
```

*(adjust based on your dataset structure)*

---

## Training

Example:

```bash
python src/train.py \
  --model resnet50 \
  --image_size 224 \
  --train_fraction 1.0
```

Options:

* `--model`: `resnet50` or `mobilenet_v2`
* `--image_size`: e.g. 128 or 224
* `--train_fraction`: e.g. 0.3 or 1.0

---

## Evaluation

```bash
python src/evaluate.py \
  --model resnet50 \
  --image_size 224 \
  --train_fraction 1.0
```

---

## Outputs

When running the project:

* `outputs/` is created automatically --> stores trained models (`.pth`)
* `results/` is created automatically --> stores metrics and confusion matrices

If these folders do not exist, they will be generated during execution.

---

## Results Summary

* ResNet50: ~76% accuracy
* MobileNetV2: ~73% accuracy

Key observations:

* Larger datasets improve performance significantly
* Smaller image sizes reduce accuracy
* ResNet performs better, MobileNet is more efficient

---

## Notes

* Dataset is excluded due to size
* Trained models are excluded
* Designed as a proof-of-concept experiment

---

## Author

Ansor Tagaev