import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import time

# Préparation des données CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)

# SVD Factorisation avec gestion du stride
class SVDConv2d(nn.Module):
    def __init__(self, conv_layer: nn.Conv2d, rank: int = None):
        super().__init__()
        stride = conv_layer.stride
        padding = conv_layer.padding
        weight = conv_layer.weight.data
        out_channels, in_channels, kH, kW = weight.shape
        weight_2d = weight.view(out_channels, -1)
        U, S, Vh = torch.linalg.svd(weight_2d, full_matrices=False)

        if rank is None or rank > S.size(0):
            rank = S.size(0)

        U_r = U[:, :rank]
        S_r = S[:rank]
        Vh_r = Vh[:rank, :]

        # CORRECTION ICI
        first_weight = (S_r.unsqueeze(1) * Vh_r).view(rank, in_channels, kH, kW)
        second_weight = U_r.view(out_channels, rank, 1, 1)

        self.first = nn.Conv2d(in_channels, rank, kernel_size=(kH, kW), padding=padding, stride=stride, bias=False)
        self.second = nn.Conv2d(rank, out_channels, kernel_size=1, stride=1, bias=False)

        self.first.weight.data = first_weight
        self.second.weight.data = second_weight

    def forward(self, x):
        return self.second(self.first(x))

def factorize_model_svd(model, rank=None):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.kernel_size == (3, 3) and module.groups == 1:
            svd_conv = SVDConv2d(module, rank)
            parent = model
            path = name.split('.')
            for p in path[:-1]:
                parent = getattr(parent, p)
            setattr(parent, path[-1], svd_conv)
    return model

def train_model(model, trainloader, testloader, device, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model.to(device)

    train_losses, train_accs, test_losses, test_accs = [], [], [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        train_loss = running_loss / len(trainloader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        test_loss, test_acc = evaluate_model_detailed(model, testloader, device, criterion)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    return train_losses, train_accs, test_losses, test_accs

def evaluate_model_detailed(model, loader, device, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return running_loss / len(loader), 100. * correct / total

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Chargement du modèle de base
from models import resnet

assert torch.cuda.is_available(), "Aucun GPU CUDA détecté."
device = torch.device('cuda')

os.makedirs("plot_results", exist_ok=True)

# Stockage des résultats
ranks = [None, 5, 10, 20, 30, 64]
params_list = []
acc_list = []
train_losses_dict = {}
test_losses_dict = {}
train_accs_dict = {}
test_accs_dict = {}
train_times_dict = {}

for rank in ranks:
    label = "Base" if rank is None else f"Rank {rank}"
    print(f"\n--- Entraînement {label} ---")
    model = resnet.ResNet18()
    if rank is not None:
        model = factorize_model_svd(model, rank=rank)

    params = count_parameters(model)
    params_list.append(params)

    start_time = time.time()
    train_losses, train_accs, test_losses, test_accs = train_model(model, trainloader, testloader, device)
    elapsed_time = time.time() - start_time
    train_times_dict[rank if rank is not None else 0] = elapsed_time

    acc_list.append(test_accs[-1])
    train_losses_dict[rank if rank is not None else 0] = train_losses
    test_losses_dict[rank if rank is not None else 0] = test_losses
    train_accs_dict[rank if rank is not None else 0] = train_accs
    test_accs_dict[rank if rank is not None else 0] = test_accs

# Visualisations
plt.figure()
plt.plot(params_list, acc_list, 'o-')
plt.xlabel('Nombre de paramètres')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs Nombre de paramètres après entraînement')
plt.grid(True)
plt.savefig("plot_results/accuracy_vs_params.png")
plt.close()

plt.figure()
for rank in train_losses_dict:
    plt.plot(train_losses_dict[rank], label=f'Train rank={rank}')
    plt.plot(test_losses_dict[rank], linestyle='--', label=f'Test rank={rank}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss par epoch (train/test)')
plt.legend()
plt.grid(True)
plt.savefig("plot_results/loss_per_epoch.png")
plt.close()

plt.figure()
for rank in train_accs_dict:
    plt.plot(train_accs_dict[rank], label=f'Train rank={rank}')
    plt.plot(test_accs_dict[rank], linestyle='--', label=f'Test rank={rank}')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy par epoch (train/test)')
plt.legend()
plt.grid(True)
plt.savefig("plot_results/accuracy_per_epoch.png")
plt.close()

plt.figure()
ranks_display = list(train_times_dict.keys())
times_display = list(train_times_dict.values())
plt.bar(ranks_display, times_display)
plt.xlabel('Rank')
plt.ylabel('Temps d\'entraînement (s)')
plt.title('Temps d\'entraînement par modèle')
plt.grid(True)
plt.savefig("plot_results/train_time_per_model.png")
plt.close()

compression_ratios = [p / params_list[0] * 100 for p in params_list]
plt.figure()
plt.plot(compression_ratios, acc_list, 'o-')
plt.xlabel('Compression (%)')
plt.ylabel('Accuracy (%)')
plt.title('Compression vs Accuracy')
plt.grid(True)
plt.savefig("plot_results/compression_vs_accuracy.png")
plt.close()

# Calcul de l'empreinte mémoire approximative en Mo (float32 = 4 octets par paramètre)
memory_mb = [p * 4 / (1024 ** 2) for p in params_list]

plt.figure()
plt.plot(params_list, memory_mb, 'o-')
plt.xlabel("Number of parameters")
plt.ylabel("Memory Footprint (MB)")
plt.title("Memory Footprint (MB) according to the number of parameters")
plt.grid(True)
plt.savefig("plot_results/memory_footprint_vs_parameters.png")
plt.close()

# Affichage de la taille du modèle
model_size_bytes = os.path.getsize("pth_files/resnet18_factorized.pth")
model_size_mb = model_size_bytes / (1024 * 1024)
print(f"📦 Model Size on Disk: {model_size_mb:.2f} MB")