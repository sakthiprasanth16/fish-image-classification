# 🐟 Multiclass Fish Image Classification

## 📌 Project Overview
This project implements a complete **end-to-end deep learning pipeline** for **multiclass fish image classification**. It includes data preprocessing, handling class imbalance through augmentation, training a CNN from scratch, experimenting with multiple transfer learning architectures, evaluating model performance, and deploying the best model using **Streamlit**.

**Best Performing Model:** EfficientNetB0 (~91% accuracy)  
**Fastest Inference Model:** MobileNet  

---

## 📂 Dataset
The dataset follows a directory-per-class structure compatible with `flow_from_directory`:

data/
├── train/
├── val/
└── test/


- Some classes are highly imbalanced (e.g., `animal fish bass`)
- Targeted image augmentation was applied to balance smaller classes

**Dataset Link:**  
https://drive.google.com/drive/folders/1iKdOs4slf3XvNWkeSfsszhPRggfJ2qEd

---

## 🧹 Data Preprocessing & Augmentation

- Image size: **256 × 256**
- Color mode: **RGB**
- Rescaling: `1./255` (for Custom CNN)
- Backbone-specific `preprocess_input` for transfer learning models
- Augmentation techniques:
  - Rotation
  - Width & height shift
  - Zoom
  - Shear
  - Horizontal flip

---

## 🧠 Models

### Custom CNN (Baseline)
- 3 Convolution layers with MaxPooling
- Dropout for regularization
- Dense layers with Softmax output
- Accuracy: **~78%**

### Transfer Learning Models
- VGG16
- ResNet50
- MobileNet
- InceptionV3
- EfficientNetB0

**Fine-Tuning Strategy:**
- Freeze pretrained backbone
- Train classifier head
- Unfreeze last layers
- Train with lower learning rate

---

## 📈 Training & Evaluation

### Hyperparameters
- Image size: `256 × 256`
- Batch size: `32`
- Optimizer: `Adam`
- Loss: `categorical_crossentropy`
- Learning rate:
  - Head training: `1e-3`
  - Fine-tuning: `1e-5`

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## 🚀 Deployment (Streamlit)

This project includes a **Streamlit web application** for real-time fish species classification using the **EfficientNetB0 fine-tuned model**.

---

### 🔹 Local Deployment

**Step 1: Install dependencies**
```bash
pip install -r requirements.txt

Step 2: Run the Streamlit app
streamlit run app.py

Step 3: Open the app in your browser
http://localhost:8501

---

## 🧠 Simple Memory Trick
- **Open code block** → ```bash  
- **Close code block** → ```  
- Then write normal text

---

## ❌ WRONG (What you did)
```markdown
```bash
pip install -r requirements.txt
Step 2 text here   ❌ (still inside code block)

---

### 🔹 Google Colab Deployment (Open Streamlit in Colab)

Streamlit apps running on Google Colab are not accessible directly.
To open the app in a browser, we use Cloudflare Tunnel (cloudflared).
Step 1: Install Cloudflared

wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
chmod +x cloudflared
mv cloudflared /usr/local/bin/


Step 2: Create the Streamlit app

%%writefile app.py
# Streamlit app code using EfficientNetB0 fine-tuned model


Step 3: Run Streamlit in background

streamlit run app.py &>/dev/null &


Step 4: Expose Streamlit app and get public URL

cloudflared tunnel --url http://localhost:8501


After running this command, a public HTTPS URL will be displayed in the output.
Open that URL in any browser to access the Streamlit application.


