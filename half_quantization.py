import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models import resnet  # Ton import personnalisé

# Fonction utilitaire pour convertir seulement les BatchNorm en float32
def convert_batchnorm_to_float(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        convert_batchnorm_to_float(child)

# Configuration du device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Forçage du déterminisme pour reproductibilité
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Prétraitement
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Chargement des données
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=False, num_workers=4)

# Chemin du modèle sauvegardé
PATH = 'pth_files/resnet18_cifar10_best.pth'

# Évaluation du modèle float32
model = resnet.ResNet18().to(device)
model.load_state_dict(torch.load(PATH))
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

acc_float = 100 * correct / total
print(f'Accuracy of the pretrained ResNet on the test set (Float32): {acc_float:.2f} %')

# Évaluation du modèle quantifié en half precision (Float16)
model = resnet.ResNet18().to(device)
model.load_state_dict(torch.load(PATH))
model.half()
convert_batchnorm_to_float(model)  # Important pour la stabilité
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images = images.to(device).half()
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

acc_half = 100 * correct / total
print(f'Accuracy of the post-training quantized to half pretrained model (Float16): {acc_half:.2f} %')

# Visualisation
plt.figure(figsize=(6, 4))
plt.bar(['Float32', 'Float16'], [acc_float, acc_half], color=['blue', 'green'])
plt.title('Accuracy Comparison: Float32 vs Half Precision')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("plot_results/half_and_float.png")  # Sauvegarde avant plt.show()
plt.show()

import time

# Fonction de mesure du temps d'inférence
def measure_inference_time(model, dataloader, device, precision='float32', num_batches=50):
    model.eval()
    times = []
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            if precision == 'float16':
                images = images.to(device).half()
            else:
                images = images.to(device)

            torch.cuda.synchronize()  # Synchronisation avant mesure
            start_time = time.time()
            
            outputs = model(images)
            
            torch.cuda.synchronize()  # Synchronisation après inférence
            end_time = time.time()
            
            times.append(end_time - start_time)
    return sum(times) / len(times)

# Chargement du modèle float32
model_float32 = resnet.ResNet18().to(device)
model_float32.load_state_dict(torch.load(PATH))

# Chargement du modèle float16
model_float16 = resnet.ResNet18().to(device)
model_float16.load_state_dict(torch.load(PATH))
model_float16.half()
convert_batchnorm_to_float(model_float16)

# Mesure du temps moyen d'inférence
time_float32 = measure_inference_time(model_float32, testloader, device, precision='float32')
time_float16 = measure_inference_time(model_float16, testloader, device, precision='float16')

print(f"\nAverage inference time per batch (Float32) : {time_float32*1000:.2f} ms")
print(f"Average inference time per batch (Float16) : {time_float16*1000:.2f} ms")

import matplotlib.pyplot as plt

# Suppose que tu as déjà ces valeurs mesurées :
# acc_float, acc_half
# time_float32, time_float16
# memory_float32, memory_float16
# Fonction pour mesurer la mémoire GPU
def measure_memory_usage(model, dataloader, device, precision='float32', num_batches=10):
    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            if precision == 'float16':
                images = images.to(device).half()
            else:
                images = images.to(device)
            _ = model(images)  # Juste un passage pour allouer la mémoire
    torch.cuda.synchronize()
    memory_allocated = torch.cuda.max_memory_allocated(device) / (1024**2)  # En MB
    torch.cuda.reset_peak_memory_stats(device)
    return memory_allocated

# Chargement des modèles
model_float32 = resnet.ResNet18().to(device)
model_float32.load_state_dict(torch.load(PATH))

model_float16 = resnet.ResNet18().to(device)
model_float16.load_state_dict(torch.load(PATH))
model_float16.half()
convert_batchnorm_to_float(model_float16)

# Mesure de la mémoire
memory_float32 = measure_memory_usage(model_float32, testloader, device, precision='float32')
memory_float16 = measure_memory_usage(model_float16, testloader, device, precision='float16')

print(f"\nPeak memory usage per batch (Float32) : {memory_float32:.2f} MB")
print(f"Peak memory usage per batch (Float16) : {memory_float16:.2f} MB")

# Données
categories = ['Float32', 'Float16']

accuracies = [acc_float, acc_half]
times = [time_float32 * 1000, time_float16 * 1000]  # En millisecondes
memories = [memory_float32, memory_float16]  # En MB

# Création de la figure
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Plot Accuracy
axs[0].bar(categories, accuracies)
axs[0].set_title('Accuracy (%)')
axs[0].set_ylabel('Accuracy (%)')
axs[0].set_ylim(0, 100)
axs[0].grid(axis='y', linestyle='--', alpha=0.7)

# Plot Temps d'inférence
axs[1].bar(categories, times)
axs[1].set_title('Inference Time per Batch (ms)')
axs[1].set_ylabel('Time (ms)')
axs[1].grid(axis='y', linestyle='--', alpha=0.7)

# Plot Mémoire GPU
axs[2].bar(categories, memories)
axs[2].set_title('Peak Memory Usage (MB)')
axs[2].set_ylabel('Memory (MB)')
axs[2].grid(axis='y', linestyle='--', alpha=0.7)

# Finalisation
plt.suptitle('Comparison: Float32 vs Float16', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("plot_results/comparison_float32_vs_float16.png")
plt.show()

import os

# Taille du fichier du modèle sauvegardé
model_path = 'pth_files/resnet18_cifar10_best.pth'
model_size_bytes = os.path.getsize(model_path)
model_size_mb = model_size_bytes / (1024 * 1024)

# Affichage
print(f"📦 Model Size on Disk: {model_size_mb:.2f} MB")

# Visualisation
plt.figure()
plt.bar(['Float32 Model Size'], [model_size_mb])
plt.ylabel('Size (MB)')
plt.title('Model Size')
plt.tight_layout()
plt.savefig("plot_results/model_size.png")
plt.show()

# Estimation simplifiée du débit : nombre de paramètres * bits / MB
num_params = sum(p.numel() for p in model.parameters())
bits_per_weight_float32 = 32
bits_per_weight_float16 = 16

throughput_float32_mbit = num_params * bits_per_weight_float32 / 1e6
throughput_float16_mbit = num_params * bits_per_weight_float16 / 1e6

plt.figure()
plt.bar(['Float32', 'Float16'], [throughput_float32_mbit, throughput_float16_mbit])
plt.ylabel('Throughput (Mbit)')
plt.title('Throughput vs Precision')
plt.tight_layout()
plt.savefig("plot_results/throughput_vs_precision.png")
plt.show()

print(f"Estimated Throughput (Float32): {throughput_float32_mbit:.2f} Mbit")
print(f"Estimated Throughput (Float16): {throughput_float16_mbit:.2f} Mbit")


