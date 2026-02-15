import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.resnet18 import ResNet18_CIFAR
from PIL import Image
import numpy as np


#  Configuration

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.75
MAX_FAILURES = 20

MODEL_PATH = "experiments/baseline/best_model.pth"
SAVE_DIR = "failure_cases/baseline"

os.makedirs(SAVE_DIR, exist_ok=True)

# CIFAR-10 label names
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


#  Dataset & Loader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=False, transform=transform
)

test_loader = DataLoader(
    test_dataset, batch_size=1, shuffle=False, num_workers=0
)


#  Load Model

model = ResNet18_CIFAR(num_classes=10)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


#  Failure Extraction

failure_count = 0

with torch.no_grad():
    for idx, (image, label) in enumerate(test_loader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)

        logits = model(image)
        probs = F.softmax(logits, dim=1)

        confidence, pred = torch.max(probs, dim=1)

        pred = pred.item()
        true = label.item()
        confidence = confidence.item()

        # Check for confident failure
        if pred != true and confidence >= CONFIDENCE_THRESHOLD:
            failure_count += 1
            case_dir = os.path.join(SAVE_DIR, f"case_{failure_count:02d}")
            os.makedirs(case_dir, exist_ok=True)

            # Save image (de-normalize)
            img = image.cpu().squeeze(0)
            img = img * torch.tensor((0.2470, 0.2435, 0.2616)).view(3,1,1)
            img = img + torch.tensor((0.4914, 0.4822, 0.4465)).view(3,1,1)
            img = torch.clamp(img, 0, 1)

            img = transforms.ToPILImage()(img)
            img.save(os.path.join(case_dir, "image.png"))

            # Save metadata
            with open(os.path.join(case_dir, "meta.txt"), "w") as f:
                f.write(f"Image index: {idx}\n")
                f.write(f"True label: {CLASS_NAMES[true]} ({true})\n")
                f.write(f"Predicted label: {CLASS_NAMES[pred]} ({pred})\n")
                f.write(f"Confidence: {confidence:.4f}\n")

            print(
                f"[Saved] Case {failure_count:02d} | "
                f"True: {CLASS_NAMES[true]} | "
                f"Pred: {CLASS_NAMES[pred]} | "
                f"Conf: {confidence:.2f}"
            )

        if failure_count >= MAX_FAILURES:
            break

print(f"\nTotal confident failures saved: {failure_count}")
