import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models import resnet
import binaryconnect
import torch.nn.utils.prune as prune
import pickle

# Détection du device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparamètres
batch_size = 32
lr = 0.0358408115435323
weight_decay = 0.002345625688196673
step_size = 12
gamma = 0.28261595872305967
n_epochs = 28
pruning_ratio = 0.2  # Le ratio de pruning pour pruning structuré

# Prétraitement des données
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Chargement des datasets
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

# Initialisation du modèle BC
mymodel = resnet.ResNet18()
mymodelbc = binaryconnect.BC(mymodel)
mymodelbc.model = mymodelbc.model.to(device)

# Pruning structuré du modèle (avant l'entraînement)
def prune_model(model, pruning_ratio=0.2, structured=True):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if structured:
                prune.ln_structured(module, name='weight', amount=pruning_ratio, dim=0, n=1)
                prune.remove(module, 'weight')  # Applique définitivement le pruning structuré
        elif isinstance(module, nn.Linear):
            if structured:
                prune.ln_structured(module, name='weight', amount=pruning_ratio, dim=0, n=1)
                prune.remove(module, 'weight')  # Applique définitivement le pruning structuré

# Appliquer le pruning structuré avant l'entraînement
prune_model(mymodelbc.model, pruning_ratio=pruning_ratio, structured=True)

# Loss, optimiseur et scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(mymodelbc.model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# Entraînement
losses = []
accuracies = []

# print('🚀 Start Training...')
# for epoch in range(5):
#     mymodelbc.model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0

#     for i, (inputs, labels) in enumerate(trainloader):
#         inputs, labels = inputs.to(device), labels.to(device)

#         mymodelbc.binarization()  # Binarisation avant chaque étape
#         optimizer.zero_grad()
#         outputs = mymodelbc.model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         mymodelbc.restore()  # Restaurer les poids après binarisation
#         optimizer.step()
#         mymodelbc.clip()  # Appliquer les contraintes de BinaryConnect

#         running_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total += labels.size(0)
#         correct += predicted.eq(labels).sum().item()

#     scheduler.step()
    
#     accuracy = 100. * correct / total
#     epoch_loss = running_loss / len(trainloader)
#     accuracies.append(accuracy)
#     losses.append(epoch_loss)
#     print(f"✅ Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Accuracy = {accuracy:.2f}%")

# # Évaluation sur le test set
# print("\n🔍 Évaluation sur le test set...")
mymodel.load_state_dict(torch.load("bc.pth"))
mymodelbc.model.eval()

mymodelbc.binarization()
test_running_loss = 0.0
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = mymodelbc.model(images)
        loss = criterion(outputs, labels)

        test_running_loss += loss.item()
        _, predicted = outputs.max(1)
        test_total += labels.size(0)
        test_correct += predicted.eq(labels).sum().item()

mymodelbc.restore()

test_accuracy = 100. * test_correct / test_total
test_loss = test_running_loss / len(testloader)

# Sauvegarde des métriques
with open('losses_accuracy_quantization.pkl', 'wb') as f:
    pickle.dump({
        'train': {
            'losses': losses, 
            'accuracies': accuracies
        },
        'test': {
            'losses': [test_loss], 
            'accuracies': [test_accuracy]
        }
    }, f)

print("Données sauvegardées dans 'losses_accuracy_quantization.pkl'.")
print(f"🎯 Accuracy du modèle BC sur le test set: {test_accuracy:.2f}%")
print(f"🎯 Loss du modèle BC sur le test set: {test_loss:.4f}")

import matplotlib.pyplot as plt
import pickle

# Charger le fichier contenant les métriques
with open("losses_accuracies_quantization.pkl", "rb") as f:
    metrics = pickle.load(f)

# Extraire les données
train_losses = metrics["train"]["losses"]
train_accuracies = metrics["train"]["accuracies"]
test_loss = metrics["test"]["losses"][0]
test_accuracy = metrics["test"]["accuracies"][0]

epochs = list(range(1, len(train_losses) + 1))

# Créer une figure avec deux sous-plots
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Training Loss Plot
axs[0].plot(epochs, train_losses, marker='o', linestyle='-')
axs[0].set_title('Training Loss', fontsize=14)
axs[0].set_xlabel('Epoch', fontsize=12)
axs[0].set_ylabel('Loss', fontsize=12)
axs[0].grid(axis='y', linestyle='--', alpha=0.7)

# Training Accuracy Plot
axs[1].plot(epochs, train_accuracies, marker='o', linestyle='-', label='Train Accuracy')
axs[1].axhline(test_accuracy, color='red', linestyle='--', label=f'Test Accuracy: {test_accuracy:.2f}%')
axs[1].set_title('Training Accuracy', fontsize=14)
axs[1].set_xlabel('Epoch', fontsize=12)
axs[1].set_ylabel('Accuracy (%)', fontsize=12)
axs[1].grid(axis='y', linestyle='--', alpha=0.7)
axs[1].legend()

# Finalisation
plt.suptitle('BinaryConnect Training - Loss and Accuracy', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("binaryconnect_training_summary.png")
plt.show()

import torch.nn as nn

# --- Fonctions auxiliaires ---
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_nonzero_parameters(model):
    return sum(p.nonzero().size(0) for p in model.parameters())

def count_macs(model, input_size=(3, 32, 32)):
    dummy_input = torch.randn(1, *input_size).to(device)
    macs = 0
    hooks = []

    def count_mac_hook(module, input, output):
        nonlocal macs  # <= CORRECTION ici
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            mac = output.nelement() * input[0].size(1)
            macs += mac

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(count_mac_hook))

    model.eval()
    with torch.no_grad():
        model(dummy_input)

    for h in hooks:
        h.remove()

    return macs


def compute_score(model, pruning_ratio, quantization_ratio, params, ops):
    p_s = pruning_ratio
    p_u = 0.0
    q_w = quantization_ratio
    q_a = quantization_ratio

    score = (1 - (p_s + p_u)) * (q_w / 32) * params / (5.6e6) + (1 - p_s) * (max(q_w, q_a) / 32) * ops / (2.8e8)
    return score

# --- Calcul et affichage final ---

# Remarque : ton modèle est encapsulé dans BinaryConnect, donc on utilise mymodelbc.model
params_total = count_parameters(mymodelbc.model)
params_nonzero = count_nonzero_parameters(mymodelbc.model)
sparsity = 100 * (1 - params_nonzero / params_total)
macs_total = count_macs(mymodelbc.model)
score_final = compute_score(mymodelbc.model, pruning_ratio=pruning_ratio, quantization_ratio=0.5, params=params_total, ops=macs_total)

print("\n📊 Résultats du modèle après Pruning + Binarization:")
print(f"Nombre total de paramètres        : {params_total:,}")
print(f"Nombre de paramètres non nuls      : {params_nonzero:,}")
print(f"Sparsité finale du modèle           : {sparsity:.2f}%")
print(f"Nombre approximatif de MACs         : {macs_total:,}")
print(f"Score final du modèle               : {score_final:.6f}")
