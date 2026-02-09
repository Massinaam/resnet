import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import numpy as np
import optuna
import pickle
import matplotlib.pyplot as plt
from models import resnet

# Prétraitement
normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

# Dataset complet
rootdir = '/opt/img/effdl-cifar10/'
c10train = CIFAR10(rootdir, train=True, download=True, transform=transform_train)
c10test = CIFAR10(rootdir, train=False, download=True, transform=transform_test)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Stockage métriques Optuna
optuna_metrics = []

# # Fonction d'entraînement (appelée par Optuna)
# def train_model(trial):
#     batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
#     lr = trial.suggest_loguniform("lr", 1e-4, 1e-1)
#     optimizer_name = trial.suggest_categorical("optimizer", ["SGD", "Adam"])
#     weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
#     step_size = trial.suggest_int("step_size", 10, 30)
#     gamma = trial.suggest_uniform("gamma", 0.1, 0.9)
#     n_epochs = trial.suggest_int("n_epochs", 10, 30)

#     trainloader = DataLoader(c10train, batch_size=batch_size, shuffle=True)
#     testloader = DataLoader(c10test, batch_size=64)

#     net = resnet.ResNet18().to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = getattr(optim, optimizer_name)(net.parameters(), lr=lr, weight_decay=weight_decay)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

#     losses, accuracies = [], []
#     for epoch in range(n_epochs):
#         net.train()
#         correct, total, running_loss = 0, 0, 0.0
#         for inputs, labels in trainloader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = net(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#             correct += (outputs.argmax(1) == labels).sum().item()
#             total += labels.size(0)
#         scheduler.step()
#         losses.append(running_loss / len(trainloader))
#         accuracies.append(correct / total)

#     # Évaluation
#     net.eval()
#     test_correct, test_total = 0, 0
#     with torch.no_grad():
#         for inputs, labels in testloader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = net(inputs)
#             test_correct += (outputs.argmax(1) == labels).sum().item()
#             test_total += labels.size(0)
#     test_acc = test_correct / test_total

#     optuna_metrics.append({
#         'trial_params': trial.params,
#         'train_losses': losses,
#         'train_accuracies': accuracies,
#         'test_accuracy': test_acc,
#         'n_epochs': n_epochs
#     })

#     return 1.0 - test_acc

# # --- OPTIMISATION OPTUNA (en mémoire) ---
# study = optuna.create_study(direction="minimize")
# study.optimize(train_model, n_trials=50)

# # Sauvegarde des résultats
# with open("optuna_results.pkl", "wb") as f:
#     pickle.dump({
#         "trials_dataframe": study.trials_dataframe(),
#         "best_params": study.best_params,
#         "metrics": optuna_metrics,
#     }, f)


# --- ENTRAÎNEMENT FINAL ---
# best_params = study.best_params
best_params = {
    'batch_size': 32, 
    'lr': 0.0358408115435323, 
    'optimizer': 'AdamW', 
    'weight_decay': 0.002345625688196673, 
    'step_size': 50, 
    'gamma': 0.28261595872305967, 
    'n_epochs': 70}


batch_size = best_params['batch_size']
lr = best_params['lr']
optimizer_name = best_params['optimizer']
weight_decay = best_params['weight_decay']
step_size = best_params['step_size']
gamma = best_params['gamma']
n_epochs = best_params['n_epochs']

trainloader = DataLoader(c10train, batch_size=batch_size, shuffle=True)
testloader = DataLoader(c10test, batch_size=64)

net = resnet.ResNet18().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = getattr(optim, optimizer_name)(net.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

losses, accuracies = [], []
test_losses = []
test_accuracies = []

for epoch in range(n_epochs):
    # Entraînement
    net.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    scheduler.step()
    losses.append(running_loss / len(trainloader))
    accuracies.append(correct / total)

    # Évaluation après chaque epoch
    net.eval()
    epoch_test_losses = []
    epoch_test_accuracies = []
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            epoch_test_losses.append(loss.item())

            correct = (outputs.argmax(1) == labels).sum().item()
            total = labels.size(0)
            epoch_test_accuracies.append(correct / total)
    
    # Stocker la moyenne par epoch
    test_loss_mean = np.mean(epoch_test_losses)
    test_acc_mean = np.mean(epoch_test_accuracies)
    print(f"Epoch {epoch+1}: Train Acc = {correct / total * 100:.2f}%, Test Acc = {test_acc_mean * 100:.2f}%")

    # Enregistrer les moyennes par epoch
    test_losses.append(test_loss_mean)
    test_accuracies.append(test_acc_mean)


# Sauvegarde
with open("pkl_files/final_training_metrics.pkl", "wb") as f:
    pickle.dump({
        'train_losses': losses,
        'train_accuracies': accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies,
        'params': best_params
    }, f)

torch.save(net.state_dict(), "pth_files/resnet18_cifar10_best.pth")

print(f"\n✅ Test accuracy with best hyperparameters: {test_acc_mean * 100:.2f}%")
print(f"✅ Test loss with best hyperparameters: {test_loss_mean:.4f}")
print(f"✅ Best hyperparameters: {best_params}")