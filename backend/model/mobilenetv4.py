DIR = "/content/drive/MyDrive/TIRADS/"
LOCALDIR  = "/content/drive/MyDrive/TIRADS/1. acc-80_MobileNetV2_PyTorch/"
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import os
from tqdm import tqdm
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants (Adjusted for CPU)
DATA_PATH = r"C:\Users\jessica\Downloads\Thyroma\backend\Thyroid_Data"
NUM_CLASSES = 5
BATCH_SIZE = 16  # Reduced for CPU memory
IMG_SIZE = 128   # Smaller image size
EPOCHS = 10      # Reduced epochs
LR = 1e-4        # Learning rate

# Simplified transforms
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
full_dataset = ImageFolder(DATA_PATH)

# Split dataset
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# Apply transforms
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform
test_dataset.dataset.transform = val_transform

# CPU-optimized dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2  # Reduced workers for CPU
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Simplified model
class ThyroidClassifier(nn.Module):
    def __init__(self):
        super(ThyroidClassifier, self).__init__()
        self.base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.base.classifier[1] = nn.Linear(1280, NUM_CLASSES)

    def forward(self, x):
        return self.base(x)

model = ThyroidClassifier()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
# from google.colab import drive
# drive.mount('/content/drive')
# # Training loop
# best_val_acc = 0
# EPOCHS = 10
# for epoch in range(EPOCHS):
#     # Training
#     model.train()
#     train_loss = 0.0
#     correct = 0
#     total = 0

#     for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
#         # images, labels = images.to(device), labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total += labels.size(0)
#         correct += predicted.eq(labels).sum().item()

#     # Validation
#     model.eval()
#     val_loss = 0.0
#     val_correct = 0
#     val_total = 0

#     with torch.no_grad():
#         for images, labels in val_loader:
#             # images, labels = images.to(device), labels.to(device)

#             outputs = model(images)
#             loss = criterion(outputs, labels)

#             val_loss += loss.item()
#             _, predicted = outputs.max(1)
#             val_total += labels.size(0)
#             val_correct += predicted.eq(labels).sum().item()

#     # Statistics
#     train_acc = 100 * correct / total
#     val_acc = 100 * val_correct / val_total

#     print(f"Epoch {epoch+1}: "
#           f"Train Loss: {train_loss/len(train_loader):.4f} | "
#           f"Train Acc: {train_acc:.2f}% | "
#           f"Val Acc: {val_acc:.2f}%")

#     scheduler.step(val_loss)

#     # Save best model
#     if val_acc > best_val_acc:
#         best_val_acc = val_acc
#         torch.save(model.state_dict(), r"C:\Users\jessica\Downloads\Thyroma\saved_models\MobileNetV2.pth")
# Load best model
model.load_state_dict(torch.load(r"C:\Users\jessica\Downloads\Thyroma\saved_models\MobileNetV2.pth"))

# CPU-optimized prediction function
def predict_tirads(image_path):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = val_transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)

    return [f"{round(float(probs[0][i])*100, 2)}%" for i in range(NUM_CLASSES)]

# Example usage
# probabilities = predict_tirads(DIR + "test_samples/sample9-unknown.jpg")
# for i in range(5):
#     print(f"TIRADS {i+1} : {probabilities[i]}%")