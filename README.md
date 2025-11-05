# 🧠 Early Detection of Alzheimer's Disease Using AI

A deep learning–based system for the **early detection of Alzheimer's Disease** from **MRI brain scans**, built using **PyTorch** and deployed with a **Streamlit web interface**.

This project demonstrates how artificial intelligence can assist in analyzing MRI scans for early-stage Alzheimer's — providing fast, interpretable, and explainable insights.

---

## 📂 Project Structure

```
alzheimer_project/
├── app/
│   └── app.py                    # Streamlit web app (main interface)
├── src/
│   ├── __init__.py
│   ├── model.py                  # CNN model architecture
│   ├── train.py                  # Training script
│   ├── gradcam.py                # Grad-CAM visualization
│   └── mri_preprocessing.py      # MRI to PNG preprocessing
├── data/
│   ├── processed/                # Organized data (Healthy / Alzheimer)
│   └── test/                     # Test samples
├── models/
│   └── alzheimer_cnn.pth         # Trained CNN weights
├── test.py                       # Model testing & evaluation
├── test_preprocessing.py         # For verifying preprocessing output
└── README.md
```

---

## 🚀 Features

✅ **MRI Preprocessing** – Converts `.hdr` / `.img` scans to 2D `.png` slices  
✅ **Deep Learning Model** – CNN trained on Alzheimer vs Healthy brain scans  
✅ **Explainability** – Integrated **Grad-CAM** visualization to show attention regions  
✅ **Streamlit App** – Simple and interactive web interface  
✅ **Modular Code** – Cleanly separated scripts for clarity and reuse

---

## 🧠 Model Overview

- **Architecture:** Custom Convolutional Neural Network (CNN)
- **Input Size:** 128 × 128 MRI slice
- **Classes:** `Alzheimer`, `Healthy`
- **Optimizer:** Adam
- **Loss Function:** CrossEntropyLoss
- **Epochs:** 10
- **Framework:** PyTorch

---

## 💻 How to Run

### 1️⃣ Clone this repository

```bash
git clone https://github.com/YOUR-USERNAME/alzheimer-ai.git
cd alzheimer-ai
```

### 2️⃣ Create a virtual environment

```bash
conda create -n alzheimer_env python=3.9
conda activate alzheimer_env
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the app

```bash
streamlit run app/app.py
```

### 5️⃣ (Optional) Train the model

```bash
python src/train.py
```

---

## 🧩 Example Output

| MRI Input | Grad-CAM Heatmap | Classification |
|-----------|------------------|----------------|
| 🧠 Brain slice | 🔥 Attention on temporal region | ✅ Healthy |
| 🧠 Brain slice | 🔥 Diffuse activation near hippocampus | ⚠️ Alzheimer |

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **Training Accuracy** | ~92% |
| **Validation Accuracy** | ~88% |
| **Testing Accuracy** | ~85% |

The Grad-CAM visualization highlights regions of the brain contributing most to the model's prediction — often aligning with areas clinically associated with Alzheimer's disease, such as the **hippocampus** and **temporal lobe**.

---

## 🧠 App Preview

**Homepage**
> Upload an MRI slice and view instant prediction results

**Prediction Output**
> Displays class (`Alzheimer` / `Healthy`) with confidence percentage

**Grad-CAM Heatmap**
> Click "Show Heatmap" to visualize the model's focus area

---

## 📘 Technologies Used

| Category | Tools |
|----------|-------|
| **Language** | Python |
| **Deep Learning** | PyTorch |
| **Web Interface** | Streamlit |
| **Visualization** | Matplotlib, OpenCV |
| **Data Source** | OASIS / Kaggle MRI Datasets |

---

## ⚠️ Disclaimer

This project is intended **for academic and research purposes only.**  
It is **not approved for clinical or diagnostic use.**

---

## 🧾 Author

**Ankit Yadav**  

## ⭐ Acknowledgements

Special thanks to:
- **OASIS & Kaggle** for providing open-access MRI datasets
- **PyTorch** & **Streamlit** communities for excellent documentation

---

## 🧰 License

This project is released under the **MIT License** — free for educational and research use.

---

## 📦 Optional: requirements.txt

```txt
torch
torchvision
streamlit
pillow
numpy
opencv-python
matplotlib
scikit-learn
```
