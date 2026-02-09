import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models import resnet
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt

# Configuration du device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prétraitement des données
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Chargement des datasets
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)

# Initialisation du modèle
model = resnet.ResNet18().to(device)

# Appliquer des méthodes de pruning
def prune_model(model, pruning_ratio=0.2):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
            prune.remove(module, 'weight')  # Applique définitivement le pruning

prune_model(model, pruning_ratio=0.2)

# Initialisation de l'optimiseur (SGD)
optimizer = optim.SGD(model.parameters(), lr=0.0358408115435323, weight_decay=0.002345625688196673)

# Scheduler de taux d'apprentissage
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.28261595872305967)

# Critère de perte
criterion = nn.CrossEntropyLoss()

# Entraînement du modèle
def train_model(model, trainloader, criterion, optimizer, scheduler, epochs=28):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Convertir les entrées en half precision si le modèle est en half precision
            if next(model.parameters()).dtype == torch.float16:
                inputs = inputs.half()  # Convertir les données d'entrée en half precision

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()  # Mise à jour du learning rate

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(trainloader):.4f}, Accuracy: {100 * correct / total:.2f}%")

train_model(model, trainloader, criterion, optimizer, scheduler, epochs=28)

# Évaluation sur le test set
def evaluate_model(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Convertir les entrées en half precision si le modèle est en half precision
            if next(model.parameters()).dtype == torch.float16:
                inputs = inputs.half()  # Convertir les données d'entrée en half precision

            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    print(f"Accuracy on test set: {100 * correct / total:.2f}%")

evaluate_model(model, testloader)

# Quantification des poids (Half Precision)
model.half()  # Quantification en half precision

# Sauvegarde du modèle après quantification et pruning
torch.save(model.state_dict(), 'pth_files/resnet18_cifar10_pruned_and_quantized.pth')