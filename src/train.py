import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import AlzheimerCNN



import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = "data/processed"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "alzheimer_cnn.pth")
BATCH_SIZE = 32
IMG_SIZE = (128, 128)
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# DATA PREPARATION
# ==========================================
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # normalize for 3 channels
])

dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)

# Split dataset
train_size = int((1 - VAL_SPLIT) * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"🧠 Dataset loaded: {len(dataset)} images")
print(f"📊 Train: {len(train_dataset)}, Validation: {len(val_dataset)}")
print(f"Classes: {dataset.classes}")

# ==========================================
# MODEL DEFINITION
# ==========================================
from src.model import AlzheimerCNN

# Initialize model
model = AlzheimerCNN().to(DEVICE)


# ==========================================
# TRAINING SETUP
# ==========================================
model = AlzheimerCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

os.makedirs(MODEL_DIR, exist_ok=True)

best_val_acc = 0.0
start_time = time.time()

# ==========================================
# TRAINING LOOP
# ==========================================
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    # Validation phase
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= len(val_loader)
    val_acc = val_correct / val_total
    scheduler.step(val_loss)

    print(f"\n📅 Epoch [{epoch+1}/{NUM_EPOCHS}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        torch.save(model.state_dict(), MODEL_PATH)
        best_val_acc = val_acc
        print(f"✅ Best model updated (Val Acc: {best_val_acc:.4f})")

end_time = time.time()
print(f"\n🏁 Training complete in {(end_time - start_time)/60:.2f} mins")
print(f"🧩 Best Validation Accuracy: {best_val_acc:.4f}")
print(f"💾 Model saved to: {MODEL_PATH}")
