# Multiclass Fish Image Classification

**Project summary**

This repository contains a complete end-to-end project for classifying fish images into multiple species using deep learning. It includes data preprocessing and augmentation, training a CNN from scratch, experimenting with five transfer-learning backbones (VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0), model evaluation, saving the best models, and deploying an interactive Streamlit web app for inference.

---

## Table of Contents

* [Project overview](#project-overview)
* [Dataset](#dataset)
* [Data preprocessing & augmentation](#data-preprocessing--augmentation)
* [Modeling](#modeling)

  * [Custom CNN (baseline)](#custom-cnn-baseline)
  * [Transfer learning backbones](#transfer-learning-backbones)
  * [Fine-tuning](#fine-tuning)
* [Training & evaluation](#training--evaluation)
* [Model saving & artifacts](#model-saving--artifacts)
* [Deployment (Streamlit)](#deployment-streamlit)
* [Environment & requirements](#environment--requirements)
* [How to reproduce (quick-start)](#how-to-reproduce-quick-start)
* [Tips, troubleshooting & notes](#tips-troubleshooting--notes)
* [License & acknowledgements](#license--acknowledgements)

---

# Project overview

Goal: build a multiclass image classifier for fish species and provide a user-friendly web app to predict the species from uploaded images. The project demonstrates building a CNN from scratch, using transfer learning + fine-tuning, data augmentation to balance small classes, evaluation with standard metrics (accuracy, precision, recall, F1, confusion matrix), and deploying the model with Streamlit.

---

# Dataset

The dataset is organized in the common `train/`, `val/`, `test/` folder-per-class structure used by `flow_from_directory`:

```
data/
  ├─ train/
  │   ├─ animal fish/
  │   ├─ animal fish bass/
  │   └─ fish sea_food .../
  ├─ val/
  └─ test/
```

Example class counts (from the run you shared):

* `animal fish`: train 1096, val 187, test 520
* `animal fish bass`: train 30, val 10, test 13
* many classes ~ 500–600 images each

Because some classes (e.g. `animal fish bass`) are very small, we used targeted augmentation to increase them to target totals.

---

# Data preprocessing & augmentation

Key settings used in the project:

* Image size: `256 x 256` (RGB)
* Rescaling / preprocessing: either `rescale=1./255` for custom CNN, or `preprocessing_function` matching the backbone's `preprocess_input` for transfer-learning models.
* Augmentation transforms: rotation, width/height shift, zoom, shear, horizontal flip, `fill_mode='nearest'`.

Example augmentation script (short):

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os

aug_dir = "data/train/animal fish bass"
target_total = 564

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# load existing images, iterate and save augmented images until target_total is reached
```

Keep augmented images local (or upload to Drive) and re-generate train/val/test splits if you change the augmentation strategy.

---

# Modeling

## Custom CNN (baseline)

A small sequential CNN was used as baseline. Example architecture:

* Conv2D(32) -> MaxPool
* Conv2D(64) -> MaxPool
* Conv2D(128) -> MaxPool
* Flatten -> Dropout(0.5) -> Dense(128) -> Dense(num_classes, softmax)

Compile with `Adam` and `categorical_crossentropy` for one-hot labels.

## Transfer learning backbones

Backbones experimented with:

* VGG16
* ResNet50
* MobileNet
* InceptionV3
* EfficientNetB0

For each backbone:

1. Load `include_top=False` and `weights='imagenet'`.
2. Freeze the base, add a global average pooling, dropout, Dense(128), Dense(num_classes, softmax).
3. Train the head for a few epochs (`EPOCHS_HEAD`) with a relatively high lr (e.g. 1e-3).
4. Fine-tune: unfreeze the last `N` layers (backbone-specific) and train with a low lr (e.g. 1e-5).

The repository includes helper functions to build `ImageDataGenerator`s using each backbone's `preprocess_input` function — this is important for correct normalization.

## Fine-tuning

* Strategy: train head until validation stabilizes, then unfreeze some backbone layers and continue training with smaller LR.
* Use callbacks: `ModelCheckpoint` (save best by `val_accuracy`), `EarlyStopping`, and `ReduceLROnPlateau`.

---

# Training & evaluation

Key hyperparameters (you can change in scripts):

* `IMG_SIZE = (256, 256)`
* `BATCH_SIZE = 32`
* `EPOCHS_HEAD = 3` (example; increase for production)
* `EPOCHS_FINETUNE = 3`
* `LR_HEAD = 1e-3`, `LR_FINETUNE = 1e-5`

Evaluation metrics:

* Accuracy
* Precision / Recall / F1-score (per-class)
* Confusion matrix (visualized with seaborn heatmap)

The `evaluate_and_report` helper loads predictions using `model.predict(generator)` and prints `classification_report` and plots the confusion matrix.

---

# Model saving & artifacts

* Saved models are stored in `SAVE_DIR` (`/content/drive/MyDrive/fish_models` in Colab). File naming convention:

  * `custom_cnn.h5`
  * `VGG16_head.h5` / `VGG16_head_finetuned.h5`
  * `EfficientNetB0_head.keras` / `EfficientNetB0_head_finetuned.keras` (EfficientNet sometimes saved in native `.keras` format)
* Class indices mapping saved as `*_class_indices.json` for consistent label decoding.

---

# Deployment (Streamlit)

A simple Streamlit app (`streamlit_app/app.py`) is included. Main features:

* File uploader for JPG / PNG
* Preprocess image using the backbone-specific `preprocess_input` (for EfficientNet you used `efficientnet.preprocess_input`)
* Display top prediction and confidence

Run locally (example):

```bash
pip install -r requirements.txt
streamlit run streamlit_app/app.py
```

To expose the app from Colab (you used `cloudflared`) the typical sequence is:

```bash
# download + make cloudflared executable (already in your Colab snippet)
cloudflared tunnel --url http://localhost:8501
# in a separate cell / terminal run:
streamlit run streamlit_app/app.py
```

**Important**: When loading large models from Google Drive in Streamlit, use the cached loader `@st.cache_resource` (Streamlit 1.12+) to avoid reloading on every interaction.

---

# Environment & requirements

Create `requirements.txt` with at least the following versions (adjust as needed):

```
tensorflow>=2.11
streamlit
numpy
pandas
matplotlib
seaborn
scikit-learn
pillow
cloudflared (optional - used only as binary in Colab)
```

For Colab users: ensure Runtime → Change runtime type → GPU.

---

# How to reproduce (quick-start)

1. Clone this repo to your VS Code / Colab workspace.
2. Upload the `data/` directory to Google Drive or keep it locally. (Do not commit to GitHub.)
3. (Optional) Run augmentation scripts to balance small classes.
4. Start training: run `scripts/train_custom.py` or `scripts/train_heads.py` (edit hyperparams at top).
5. Evaluate models with `scripts/utils.py` helpers and pick the best checkpoint.
6. Fine-tune the best model: `scripts/finetune.py`.
7. Launch the Streamlit app (`streamlit run streamlit_app/app.py`) and point cloudflared to it if you need external access.

---

# Tips, troubleshooting & notes

* **Class imbalance**: Use augmentation targeted at the minority classes or use class weights.
* **Consistent preprocessing**: When using transfer learning, ALWAYS use the backbone's `preprocess_input` for training and inference.
* **Model save format**: Some Keras models (EfficientNet) are safer saved in native `.keras` format; check compatibility when loading.
* **Memory / GPU**: Large backbones + 256×256 images require GPU memory. Reduce `BATCH_SIZE` if you run out of memory.
* **Random seeds**: Set seeds (`np.random.seed`, `tf.random.set_seed`, `random.seed`) for reproducibility, but small differences can still occur.

---

# License & acknowledgements

This project is released under the MIT License — feel free to adapt for academic or commercial use. Acknowledge the use of ImageNet-pretrained backbones and any external datasets or contributors.

---

# Contact

If you want changes to the README or to include specific code snippets from your repository, tell me which sections to expand (e.g., exact Colab cells, full augmentation script, or `requirements.txt`) and I will update the README accordingly.
