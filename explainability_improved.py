import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from models.resnet18 import ResNet18_CIFAR

# =======================
# Configuration
# =======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔥 Load IMPROVED model
MODEL_PATH = "experiments/improved/best_model.pth"

# 🔥 Use baseline failure images
BASE_FAILURE_DIR = "failure_cases/baseline"

SELECTED_CASES = ["case_03", "case_08", "case_11"]

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# =======================
# Grad-CAM Hook Class
# =======================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[:, class_idx]
        score.backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1)

        cam = F.relu(cam)
        cam = cam.squeeze().cpu().detach().numpy()
        cam = cv2.resize(cam, (32, 32))
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        return cam

# =======================
# Load Improved Model
# =======================
model = ResNet18_CIFAR(num_classes=10)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

target_layer = model.model.layer4[-1].conv2
gradcam = GradCAM(model, target_layer)

# =======================
# Run Grad-CAM
# =======================
for case in SELECTED_CASES:
    case_dir = os.path.join(BASE_FAILURE_DIR, case)
    img_path = os.path.join(case_dir, "image.png")

    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    output = model(input_tensor)
    pred_class = torch.argmax(output, dim=1).item()

    cam = gradcam.generate(input_tensor, pred_class)

    img_np = np.array(image)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.5 * heatmap + 0.5 * img_np)

    # 🔥 Save with different name
    save_path = os.path.join(case_dir, "gradcam_improved.png")
    cv2.imwrite(save_path, overlay)

    print(f"Improved Grad-CAM saved for {case}")
