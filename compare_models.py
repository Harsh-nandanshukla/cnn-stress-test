import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import csv
from models.resnet18 import ResNet18_CIFAR

# ======================
# Config
# ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASELINE_PATH = "experiments/baseline/best_model.pth"
IMPROVED_PATH = "experiments/improved/best_model.pth"

CASES = ["case_03", "case_08", "case_11"]
BASE_DIR = "failure_cases/baseline"

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# ======================
# Load Model Function
# ======================
def load_model(path):
    model = ResNet18_CIFAR(num_classes=10)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

baseline_model = load_model(BASELINE_PATH)
improved_model = load_model(IMPROVED_PATH)

# ======================
# Prepare Save Directory
# ======================
os.makedirs("experiments/comparison", exist_ok=True)
csv_path = "experiments/comparison/comparison_results.csv"

# ======================
# Comparison
# ======================
with open(csv_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "case",
        "true_label",
        "baseline_prediction",
        "baseline_confidence",
        "improved_prediction",
        "improved_confidence"
    ])

    print("\n===== Model Comparison =====\n")

    for case in CASES:
        img_path = os.path.join(BASE_DIR, case, "image.png")
        meta_path = os.path.join(BASE_DIR, case, "meta.txt")

        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # Baseline
        with torch.no_grad():
            out_base = baseline_model(input_tensor)
            prob_base = F.softmax(out_base, dim=1)
            conf_base, pred_base = torch.max(prob_base, dim=1)

        # Improved
        with torch.no_grad():
            out_imp = improved_model(input_tensor)
            prob_imp = F.softmax(out_imp, dim=1)
            conf_imp, pred_imp = torch.max(prob_imp, dim=1)

        # Extract true label
        with open(meta_path, "r") as mf:
            lines = mf.readlines()
            true_line = [l for l in lines if "True label" in l][0]
            true_label = true_line.split(":")[1].strip().split()[0]

        writer.writerow([
            case,
            true_label,
            CLASS_NAMES[pred_base.item()],
            round(conf_base.item(), 4),
            CLASS_NAMES[pred_imp.item()],
            round(conf_imp.item(), 4)
        ])

        print(f"--- {case} ---")
        print(f"True: {true_label}")
        print(f"Baseline  → {CLASS_NAMES[pred_base.item()]} ({conf_base.item():.4f})")
        print(f"Improved  → {CLASS_NAMES[pred_imp.item()]} ({conf_imp.item():.4f})")
        print()

print(f"\nComparison results saved to {csv_path}")
