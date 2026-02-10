import os
import random
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.resnet18 import ResNet18_CIFAR
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

# =======================
# 1. Fixed Random Seed
# =======================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =======================
# 2. Device
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =======================
# 3. Hyperparameters
# =======================
num_classes = 10
batch_size = 64
epochs = 20
learning_rate = 0.1

# =======================
# 4. Transforms
# =======================
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])

# =======================
# 5. Datasets & Loaders
# =======================
train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=False, transform=transform_train
)

test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=False, transform=transform_test
)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
)

test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
)

# =======================
# 6. Model
# =======================
model = ResNet18_CIFAR(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(),
    lr=learning_rate,
    momentum=0.9,
    weight_decay=5e-4
)

# =======================
# 7. Training Loop
# =======================
best_acc = 0.0
os.makedirs("experiments/baseline", exist_ok=True)
csv_path = "experiments/baseline/metrics.csv"

with open(csv_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "epoch",
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy"
    ])

for epoch in range(epochs):
    model.train()
    train_loss, correct, total = 0.0, 0, 0

    for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss /= total
    train_acc = 100.0 * correct / total

    # =======================
    # 8. Evaluation
    # =======================
    model.eval()
    test_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss /= total
    test_acc = 100.0 * correct / total

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch + 1,
            train_loss,
            train_acc,
            test_loss,
            test_acc
        ])

    print(f"Epoch {epoch+1}: "
          f"Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")

    # =======================
    # 9. Save Best Model
    # =======================
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(),
                   "experiments/baseline/best_model.pth")

print(f"\nBest Test Accuracy: {best_acc:.2f}%")

# =======================
# 10. Plot Curves
# =======================
df = pd.read_csv(csv_path)

# Accuracy plot
plt.figure()
plt.plot(df["epoch"], df["train_accuracy"], label="Train Accuracy")
plt.plot(df["epoch"], df["test_accuracy"], label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Epochs")
plt.legend()
plt.grid(True)
plt.savefig("experiments/baseline/accuracy_curve.png")
plt.close()

# Loss plot
plt.figure()
plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
plt.plot(df["epoch"], df["test_loss"], label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epochs")
plt.legend()
plt.grid(True)
plt.savefig("experiments/baseline/loss_curve.png")
plt.close()
