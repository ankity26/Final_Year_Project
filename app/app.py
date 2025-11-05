import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os, sys

# ======================================================
# ✅ PATH FIX FOR IMPORTS
# ======================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import AlzheimerCNN

# ======================================================
# 🎨 PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Alzheimer’s MRI Classifier",
    page_icon="🧠",
    layout="centered",
)

# ======================================================
# ⚙️ LOAD MODEL (with toast + spinner)
# ======================================================
MODEL_PATH = "models/alzheimer_cnn.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource(show_spinner=False)
def load_model():
    model = AlzheimerCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

with st.spinner('🧠 Loading model...'):
    model = load_model()
st.toast('✅ Model loaded successfully!')

# ======================================================
# 🌈 CUSTOM CSS FOR MODERN UI
# ======================================================
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
        font-family: "Segoe UI", sans-serif;
    }
    .stButton>button {
        background: linear-gradient(to right, #14b8a6, #2563eb);
        color: white;
        border-radius: 8px;
        padding: 0.5em 1em;
        font-weight: 600;
    }
    .stButton>button:hover {
        background: linear-gradient(to right, #0ea5e9, #14b8a6);
        transition: 0.3s;
    }
    h1 {
        text-align: center;
        margin-bottom: 0.5em;
    }
    .subtext {
        text-align: center;
        font-size: 0.9em;
        color: #9CA3AF;
        margin-bottom: 2em;
    }
    </style>
""", unsafe_allow_html=True)

# ======================================================
# 🧠 TITLE & HEADER
# ======================================================
st.markdown("<h1>🧠 Alzheimer’s Detection using CNN</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtext'>Upload an MRI slice (.png or .jpg) to predict whether it shows signs of Alzheimer's disease.</p>", unsafe_allow_html=True)

# ======================================================
# 📤 IMAGE UPLOAD
# ======================================================
uploaded_file = st.file_uploader("Upload MRI Slice", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and show image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🧩 Uploaded MRI Slice", use_container_width=True, width=350)

    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Predict
    with st.spinner("🔍 Analyzing MRI..."):
        with torch.no_grad():
            outputs = model(img_tensor)
            _, pred = torch.max(outputs, 1)
            prob = torch.softmax(outputs, dim=1)[0][pred].item()

        classes = ["Alzheimer", "Healthy"]
        predicted_class = classes[pred.item()]

    st.toast("🧠 Prediction complete!")

    # ======================================================
    # 📊 DISPLAY RESULT
    # ======================================================
    st.markdown("---")
    st.subheader("🧠 Prediction Result")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Predicted Class", value=predicted_class)
    with col2:
        st.metric(label="Confidence", value=f"{prob*100:.2f}%")

    if predicted_class == "Alzheimer":
        st.warning("⚠️ The MRI shows signs consistent with Alzheimer's disease. Please consult a neurologist for a detailed evaluation.")
    else:
        st.success("✅ The MRI appears healthy with no visible signs of Alzheimer's disease.")

    # ======================================================
    # 🔥 GRAD-CAM VISUALIZATION
    # ======================================================
    if st.button("🔍 Show Grad-CAM Heatmap"):
        with st.spinner("Generating heatmap..."):

            # Hook for gradients
            feature_maps = []
            gradients = []

            def forward_hook(module, input, output):
                feature_maps.append(output)
            def backward_hook(module, grad_input, grad_output):
                gradients.append(grad_output[0])

            target_layer = model.features[-3]
            target_layer.register_forward_hook(forward_hook)
            target_layer.register_backward_hook(backward_hook)

            # Forward + Backward
            output = model(img_tensor)
            model.zero_grad()
            class_idx = pred.item()
            output[0, class_idx].backward()

            # Process Grad-CAM
            grads = gradients[0].mean(dim=[0, 2, 3]).detach().cpu().numpy()
            fmap = feature_maps[0].detach().cpu().numpy()[0]
            cam = np.zeros(fmap.shape[1:], dtype=np.float32)

            for i, w in enumerate(grads):
                cam += w * fmap[i]

            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, (image.width, image.height))
            cam = cam / cam.max()

            # Overlay Heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            img_np = np.array(image)
            overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

            # Show result
            st.markdown("### 🔥 Grad-CAM Visualization")
            st.image(overlay, caption="Model Attention Heatmap", use_container_width=True)

# ======================================================
# 📘 SIDEBAR INFO
# ======================================================
st.sidebar.title("ℹ️ Model Information")
st.sidebar.markdown("""
**Model:** Convolutional Neural Network (CNN)  
**Framework:** PyTorch  
**Input Size:** 128×128  
**Classes:** Alzheimer, Healthy  
**Training Epochs:** 10  
**Dataset:** OASIS / Kaggle MRI Dataset  
""")

st.sidebar.markdown("---")
st.sidebar.info("🧩 This is a college project using PyTorch + Streamlit for early Alzheimer’s detection. Not for medical diagnosis.")
