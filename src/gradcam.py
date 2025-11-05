import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# ===============================
# Paths and setup
# ===============================
MODEL_PATH = "models/alzheimer_cnn.pth"
IMAGE_PATH = "data/test/alzheimer"   # pick either 'healthy' or 'alzheimer'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pick a single image from test folder
img_name = os.listdir(IMAGE_PATH)[0]
img_path = os.path.join(IMAGE_PATH, img_name)

# ===============================
# Load model (same architecture)
# ===============================
class AlzheimerCNN(nn.Module):
    def __init__(self):
        super(AlzheimerCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

model = AlzheimerCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ===============================
# Prepare image
# ===============================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

image = Image.open(img_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(DEVICE)

# ===============================
# Hook the last conv layer
# ===============================
final_conv = model.features[-3]  # 3rd conv layer
gradients = None

def save_gradient(grad):
    global gradients
    gradients = grad

final_conv.register_forward_hook(lambda m, i, o: setattr(model, "feature_maps", o))
final_conv.register_full_backward_hook(lambda m, gi, go: save_gradient(go[0]))

# ===============================
# Forward + backward pass
# ===============================
output = model(input_tensor)
pred_class = torch.argmax(output, 1).item()

# Get gradients for predicted class
model.zero_grad()
output[0, pred_class].backward()

# ===============================
# Generate Grad-CAM heatmap
# ===============================
grads = gradients.mean(dim=[0, 2, 3]).detach().cpu()
feature_maps = model.feature_maps.squeeze(0).detach().cpu()

for i in range(feature_maps.shape[0]):
    feature_maps[i] *= grads[i]

heatmap = feature_maps.mean(dim=0).detach().numpy()
heatmap = np.maximum(heatmap, 0)
heatmap /= heatmap.max()

# Overlay heatmap on original image
img = np.array(image)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

# ===============================
# Display result
# ===============================
plt.figure(figsize=(10,4))
plt.subplot(1,3,1); plt.imshow(img); plt.title("Original MRI"); plt.axis("off")
plt.subplot(1,3,2); plt.imshow(heatmap, cmap="jet"); plt.title("Grad-CAM Heatmap"); plt.axis("off")
plt.subplot(1,3,3); plt.imshow(overlay); plt.title("Overlay"); plt.axis("off")
plt.suptitle(f"Predicted: {'Alzheimer' if pred_class==0 else 'Healthy'}", fontsize=14)
plt.show()
