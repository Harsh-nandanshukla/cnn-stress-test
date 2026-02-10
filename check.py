# import torch

# print("CUDA available:", torch.cuda.is_available())
# print("GPU count:", torch.cuda.device_count())

# if torch.cuda.is_available():
#     print("GPU name:", torch.cuda.get_device_name(0))
from models.resnet18 import ResNet18_CIFAR
import torch

model = ResNet18_CIFAR(num_classes=10)
x = torch.randn(1, 3, 32, 32)
y = model(x)

print(y.shape)
