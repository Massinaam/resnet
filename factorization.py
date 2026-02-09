# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from models import resnet  # assumes ResNet18 is in models/resnet.py
# import pickle
# import os

# # Device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Data preprocessing
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)

# # ---- FACTORIZATION MODULE ----

# class DepthwiseSeparableConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
#         super().__init__()
#         self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
#                                    stride=stride, padding=padding, groups=in_channels, bias=False)
#         self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

#     def forward(self, x):
#         out = self.depthwise(x)
#         out = self.pointwise(out)
#         return out

# # Replace 3x3 convolutions with depthwise separable ones
# def factorize_resnet18(model):
#     for name, module in model.named_modules():
#         if isinstance(module, nn.Conv2d) and module.kernel_size == (3, 3) and module.groups == 1:
#             depthwise_separable = DepthwiseSeparableConv(
#                 module.in_channels, module.out_channels,
#                 kernel_size=3, padding=1, stride=module.stride[0]
#             )
#             parent = model
#             path = name.split('.')
#             for p in path[:-1]:
#                 parent = getattr(parent, p)
#             setattr(parent, path[-1], depthwise_separable)
#     return model

# # Initialize and factorize ResNet18
# model = resnet.ResNet18()
# model = factorize_resnet18(model)
# model = model.to(device)

# # Optimizer and scheduler
# optimizer = optim.AdamW(model.parameters(), lr=0.0358408115435323, weight_decay=0.002345625688196673)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.28261595872305967)

# # Loss function
# criterion = nn.CrossEntropyLoss()

# # Metrics storage
# metrics = {
#     "train_loss": [],
#     "train_acc": [],
#     "test_loss": [],
#     "test_acc": []
# }

# # Training function
# def train(model, loader):
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     for inputs, targets in loader:
#         inputs, targets = inputs.to(device), targets.to(device)
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         _, predicted = outputs.max(1)
#         correct += predicted.eq(targets).sum().item()

#     acc = 100. * correct / len(loader.dataset)
#     avg_loss = running_loss / len(loader)
#     print(f"Train Loss: {avg_loss:.3f}, Accuracy: {acc:.2f}%")
#     return avg_loss, acc

# # Testing function with loss tracking
# def test(model, loader):
#     model.eval()
#     correct = 0
#     running_loss = 0.0
#     with torch.no_grad():
#         for inputs, targets in loader:
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             running_loss += loss.item()
#             _, predicted = outputs.max(1)
#             correct += predicted.eq(targets).sum().item()
#     acc = 100. * correct / len(loader.dataset)
#     avg_loss = running_loss / len(loader)
#     print(f"Test Loss: {avg_loss:.3f}, Accuracy: {acc:.2f}%")
#     return avg_loss, acc

# # ---- RUN TRAINING ----
# for epoch in range(70):
#     print(f"Epoch {epoch + 1}")
#     train_loss, train_acc = train(model, trainloader)
#     test_loss, test_acc = test(model, testloader)
#     scheduler.step()

#     metrics["train_loss"].append(train_loss)
#     metrics["train_acc"].append(train_acc)
#     metrics["test_loss"].append(test_loss)
#     metrics["test_acc"].append(test_acc)

# # Save model and metrics
# os.makedirs("pth_files", exist_ok=True)
# torch.save(model.state_dict(), "pth_files/resnet18_factorized.pth")

# os.makedirs("pkl_files", exist_ok=True)
# with open("pkl_files/resnet18_factorized_metrics.pkl", "wb") as f:
#     pickle.dump(metrics, f)
# import matplotlib.pyplot as plt

import pickle
import matplotlib.pyplot as plt
import torch
import os
from models import resnet  # suppose que ResNet18 est dans models/resnet.py

# Création du dossier de sortie pour les figures
os.makedirs("figures", exist_ok=True)

# Chargement des métriques
with open("pkl_files/resnet18_factorized_metrics.pkl", "rb") as f:
    metrics = pickle.load(f)

# Chargement du modèle factorisé pour compter les paramètres
model = resnet.ResNet18()
model.load_state_dict(torch.load("pth_files/resnet18_factorized.pth", map_location="cpu"))

from models import resnet
from factorization_module import factorize_resnet18  # ton module de factorisation

# Instancier ResNet18
model = resnet.ResNet18()

# Appliquer la factorisation (remplace les conv 3x3 par DepthwiseSeparableConv)
model = factorize_resnet18(model)

# Charger les poids du modèle factorisé
model.load_state_dict(torch.load("pth_files/resnet18_factorized.pth", map_location="cpu"))

# Calcul de la taille mémoire en MB
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
memory_bytes = params * 4  # float32 = 4 bytes
memory_mb = memory_bytes / (1024 ** 2)

# Courbes de performances
epochs = range(1, len(metrics["train_acc"]) + 1)

plt.figure()
plt.plot(epochs, metrics["train_acc"], label="Train Accuracy")
plt.plot(epochs, metrics["test_acc"], label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy over Epochs")
plt.legend()
plt.grid()
plt.savefig("figures/accuracy_over_epochs.png")
plt.close()

plt.figure()
plt.plot(epochs, metrics["train_loss"], label="Train Loss")
plt.plot(epochs, metrics["test_loss"], label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()
plt.grid()
plt.savefig("figures/loss_over_epochs.png")
plt.close()

# Graphique final : Accuracy vs Empreinte mémoire
final_test_acc = metrics["test_acc"][-1]
fig, ax1 = plt.subplots()

ax1.bar(["ResNet18 Factorized"], [final_test_acc], label="Test Accuracy (%)")
ax1.set_ylabel("Test Accuracy (%)", color="blue")

ax2 = ax1.twinx()
ax2.bar(["ResNet18 Factorized"], [memory_mb], color="orange", alpha=0.5, label="Memory (MB)")
ax2.set_ylabel("Memory Footprint (MB)", color="orange")

plt.title(f"Final Accuracy: {final_test_acc:.2f}%, Memory: {memory_mb:.2f} MB")
plt.grid()
plt.savefig("figures/final_accuracy_vs_memory.png")
plt.close()
