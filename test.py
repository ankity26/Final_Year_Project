import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# =========================================================
# CONFIGURATION
# =========================================================
# 👇 Change this line ONLY if your test folder is in another location
DATA_DIR = "data/test"
MODEL_PATH = "models/alzheimer_cnn.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# VERIFY FOLDER STRUCTURE
# =========================================================
if not os.path.exists(DATA_DIR):
    print(f"❌ Test folder not found at: {DATA_DIR}")
    print("\nPlease make sure your structure looks like this:")
    print("data/")
    print(" ├── processed/")
    print(" │    ├── healthy/")
    print(" │    └── alzheimer/")
    print(" └── test/")
    print("      ├── healthy/")
    print("      └── alzheimer/")
    raise SystemExit

if len(os.listdir(DATA_DIR)) == 0:
    print("❌ Your test folder is empty.")
    raise SystemExit

print(f"✅ Found test folder: {DATA_DIR}")
print(f"Subfolders: {os.listdir(DATA_DIR)}")

# =========================================================
# IMAGE TRANSFORMS
# =========================================================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# =========================================================
# LOAD TEST DATA
# =========================================================
test_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"🧩 Loaded {len(test_dataset)} test images from {DATA_DIR}")
print(f"Detected classes: {test_dataset.classes}")

# =========================================================
# DEFINE MODEL (same as training)
# =========================================================
from src.model import AlzheimerCNN

# Load trained model
model = AlzheimerCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# =========================================================
# LOAD TRAINED MODEL
# =========================================================
model = AlzheimerCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"✅ Loaded trained model from {MODEL_PATH}")

# =========================================================
# EVALUATE ON TEST DATA
# =========================================================
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# =========================================================
# METRICS
# =========================================================
print("\n📊 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=test_dataset.classes,
            yticklabels=test_dataset.classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Alzheimer MRI Classification")
plt.tight_layout()
plt.show()

# =========================================================
# SAMPLE PREDICTIONS
# =========================================================
import random
fig, axes = plt.subplots(2, 3, figsize=(10, 7))
for ax in axes.flat:
    idx = random.randint(0, len(test_dataset) - 1)
    image, label = test_dataset[idx]
    output = model(image.unsqueeze(0).to(DEVICE))
    pred = torch.argmax(output, 1).item()
    ax.imshow(np.transpose(image.numpy(), (1, 2, 0)))
    ax.set_title(f"True: {test_dataset.classes[label]}\nPred: {test_dataset.classes[pred]}")
    ax.axis("off")
plt.show()
