# 🐟 Multiclass Fish Image Classification

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

## Project overview

**Goal**: Build a multiclass image classifier for fish species and provide a user-friendly web app to predict the species from uploaded images. The project demonstrates building a CNN from scratch, using transfer learning + fine-tuning, data augmentation to balance small classes, evaluation with standard metrics (accuracy, precision, recall, F1, confusion matrix), and deploying the model with Streamlit.

**Project Implementation**:
- **Data Preprocessing**: Handled class imbalance through targeted augmentation
- **Model Training**: Built CNN from scratch and experimented with 5 transfer learning models
- **Evaluation**: Comprehensive metrics comparison across all models
- **Deployment**: Streamlit web application with real-time predictions

**Skills Demonstrated**: Deep Learning, Computer Vision, TensorFlow/Keras, Data Preprocessing, Transfer Learning, Model Evaluation, Streamlit Deployment

**Domain**: Image Classification / Marine Biology

---

## Dataset

The dataset is organized in the common `train/`, `val/`, `test/` folder-per-class structure used by `flow_from_directory`:
data/
├─ train/
│ ├─ animal fish/
│ ├─ animal fish bass/
│ └─ fish sea_food .../
├─ val/
└─ test/

**Dataset Statistics**:
- `animal fish`: train 1096, val 187, test 520
- `animal fish bass`: train 30, val 10, test 13 (augmented to balance)
- Many classes ~ 500–600 images each

**Dataset Link**: [Google Drive](https://drive.google.com/drive/folders/1iKdOs4slf3XvNWkeSfsszhPRggfJ2qEd?usp=sharing)

**Class Imbalance**: Some classes (e.g., `animal fish bass`) are very small, so we used targeted augmentation to increase them to target totals.

---

## Data preprocessing & augmentation

**Key Settings**:
- Image size: `256 x 256` (RGB)
- Rescaling: `rescale=1./255` for custom CNN
- Preprocessing: Backbone-specific `preprocess_input` for transfer learning models
- Augmentation transforms: rotation, width/height shift, zoom, shear, horizontal flip, `fill_mode='nearest'`

**Augmentation Example**:
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
# Load existing images, iterate and save augmented images until target_total is reached

Note: Keep augmented images local (or upload to Drive) and re-generate train/val/test splits if you change the augmentation strategy.
Modeling
Custom CNN (baseline)
A small sequential CNN was used as baseline. Architecture:

Conv2D(32) -> MaxPool

Conv2D(64) -> MaxPool

Conv2D(128) -> MaxPool

Flatten -> Dropout(0.5) -> Dense(128) -> Dense(num_classes, softmax)

Compiled with Adam optimizer and categorical_crossentropy loss for one-hot labels.

Transfer learning backbones
Models experimented with:

VGG16

ResNet50

MobileNet

InceptionV3

EfficientNetB0

Implementation Steps for each backbone:

Load include_top=False and weights='imagenet'

Freeze the base, add: GlobalAveragePooling2D -> Dropout -> Dense(128) -> Dense(num_classes, softmax)

Train the head for a few epochs with relatively high learning rate (e.g., 1e-3)

Fine-tune: unfreeze the last N layers (backbone-specific) and train with low learning rate (e.g., 1e-5)

Important: Use backbone-specific preprocess_input functions in ImageDataGenerator for correct normalization.

Fine-tuning
Strategy: Train head until validation stabilizes, then unfreeze some backbone layers and continue training with smaller learning rate.

Callbacks used:

ModelCheckpoint (save best by val_accuracy)

EarlyStopping

ReduceLROnPlateau

Training & evaluation
Key Hyperparameters:

IMG_SIZE = (256, 256)

BATCH_SIZE = 32

EPOCHS_HEAD = 3 (example; increase for production)

EPOCHS_FINETUNE = 3

LR_HEAD = 1e-3, LR_FINETUNE = 1e-5

Evaluation Metrics:

Accuracy

Precision / Recall / F1-score (per-class)

Confusion matrix (visualized with seaborn heatmap)

Evaluation Process: The evaluate_and_report helper loads predictions using model.predict(generator) and prints classification_report and plots the confusion matrix.

Performance Summary:

Best Model: EfficientNetB0 (highest accuracy ~91%)

Fastest Inference: MobileNet (optimal for deployment)

Baseline: Custom CNN (~78% accuracy, fastest training)

Model saving & artifacts
Saved Models Location: SAVE_DIR (/content/drive/MyDrive/fish_models in Colab)

File Naming Convention:

custom_cnn.h5

VGG16_head.h5 / VGG16_head_finetuned.h5

EfficientNetB0_head.keras / EfficientNetB0_head_finetuned.keras (EfficientNet sometimes saved in native .keras format)

Additional Artifacts:

Class indices mapping saved as *_class_indices.json for consistent label decoding

Training history plots and confusion matrices saved as PNG files

Deployment (Streamlit)
A Streamlit app (streamlit_app/app.py) is included with these features:

Main Features:

File uploader for JPG/PNG images

Preprocess image using backbone-specific preprocess_input

Display top prediction and confidence score

Clean, user-friendly interface

Run locally:
pip install -r requirements.txt
streamlit run streamlit_app/app.py

Expose from Colab using cloudflared:
# Download + make cloudflared executable
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x cloudflared-linux-amd64

# In one cell:
!./cloudflared-linux-amd64 tunnel --url http://localhost:8501

# In another cell:
!streamlit run streamlit_app/app.py

Environment & requirements
Create requirements.txt with at least the following versions:
tensorflow>=2.11
streamlit>=1.24
numpy>=1.21
pandas>=1.3
matplotlib>=3.5
seaborn>=0.11
scikit-learn>=1.0
pillow>=9.0

For Colab users: Ensure Runtime → Change runtime type → GPU for faster training.

How to reproduce (quick-start)
For Local Development:
Clone the repository
git clone https://github.com/yourusername/Multiclass-Fish-Image-Classification.git
cd Multiclass-Fish-Image-Classification
Set up environment
pip install -r requirements.txt

Prepare dataset

Download from Google Drive link

Extract to data/ directory

Run notebooks/preprocess.ipynb for augmentation (optional)

Train models

Run notebooks/Fish_model_building.ipynb sequentially

This notebook includes:

Custom CNN training

5 transfer learning experiments

Model evaluation and comparison

Deploy application

For Google Colab:
Upload notebooks to Colab

Mount Google Drive for data storage
from google.colab import drive
drive.mount('/content/drive')
Follow the same steps as local setup

Use cloudflared for external access to Streamlit app
