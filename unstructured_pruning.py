import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models import resnet
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt
from torch.amp import autocast
import torch.optim as optim

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

# Charger les poids du modèle sauvegardé
model.load_state_dict(torch.load("comb.pth"))
model = model.half().to(device)  # Appliquer la quantification en half precision

# Pruning structuré et non structuré du modèle
def prune_model(model, pruning_ratio=0.2, structured=True):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if structured:
                # Pruning structuré : suppression de certains filtres
                prune.ln_structured(module, name='weight', amount=pruning_ratio, dim=0, n=1)
                prune.remove(module, 'weight')  # Applique définitivement le pruning structuré
            else:
                # Pruning non structuré : suppression de certains poids
                prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
                prune.remove(module, 'weight')  # Applique définitivement le pruning non structuré
        elif isinstance(module, nn.Linear):
            if structured:
                # Pruning structuré : suppression de certains neurones
                prune.ln_structured(module, name='weight', amount=pruning_ratio, dim=0, n=1)
                prune.remove(module, 'weight')  # Applique définitivement le pruning structuré
            else:
                # Pruning non structuré : suppression de certains poids
                prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
                prune.remove(module, 'weight')  # Applique définitivement le pruning non structuré

# Appliquer le pruning structuré et non structuré sur le modèle
prune_model(model, pruning_ratio=0.2, structured=False)  # Applique le pruning structuré

# Pruning stats et plot
def print_pruning_stats(model):
    nonzero = 0
    total = 0
    for param in model.parameters():
        nonzero += param.nonzero().size(0)
        total += param.numel()
    
    sparsity = 100 * (1 - nonzero / total)
    print(f"📊 Paramètres non nuls : {nonzero:,} / {total:,}")
    print(f"📉 Sparsité du modèle : {sparsity:}%")

# Initialisation de l'optimiseur (SGD)
optimizer = optim.SGD(model.parameters(), lr=0.0358408115435323, weight_decay=0.002345625688196673)

# Scheduler de taux d'apprentissage
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.28261595872305967)

# Critère de perte
criterion = nn.CrossEntropyLoss()

for epoch in range(5):  # 5 epochs suffisent souvent
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.half()  # Très important !

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
    
    print(f"🔧 Finetuning Epoch {epoch+1}, Loss: {running_loss / len(trainloader):.4f}, Accuracy: {100 * correct / total:.2f}%")


# Fonction d'évaluation avec autocast
def evaluate_model(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            # Déplacer les entrées et les labels sur le même périphérique que le modèle
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.half()  # Convertir les données d'entrée en half precision

            # Utiliser autocast pour gérer la conversion en half precision pendant l'inférence
            with autocast(device_type='cuda', dtype=torch.half):
                outputs = model(inputs)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    print(f"Accuracy on test set: {100 * correct / total:.2f}%")

# Évaluation sur le test set
evaluate_model(model, testloader)

# Calcul du nombre de paramètres
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Calcul des opérations MACs
def count_macs(model, input_size=(3, 32, 32)):
    x = torch.ones(1, *input_size).half().to(device)  # dummy input compatible half precision
    macs = 0
    
    def count_conv_macs(module, input, output):
        nonlocal macs
        if isinstance(module, nn.Conv2d):
            height_out, width_out = output.shape[2], output.shape[3]
            kernel_size = module.kernel_size[0] * module.kernel_size[1]
            macs += (height_out * width_out) * kernel_size * module.in_channels * module.out_channels

    def count_linear_macs(module, input, output):
        nonlocal macs
        if isinstance(module, nn.Linear):
            macs += input[0].shape[1] * output.shape[1]
    
    # Enregistrer les hooks
    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hook = module.register_forward_hook(count_conv_macs if isinstance(module, nn.Conv2d) else count_linear_macs)
            hooks.append(hook)
    
    model(x)  # Passage avant (dummy forward)

    # Supprimer les hooks correctement
    for hook in hooks:
        hook.remove()

    return macs

# Calcul du score selon la formule donnée
def compute_score_unstructured(model, quantization_ratio, params, ops):
    # Calculer la sparsité réelle (non structurée)
    total = 0
    nonzero = 0
    for param in model.parameters():
        total += param.numel()
        nonzero += param.nonzero().size(0)
    
    sparsity = 1 - (nonzero / total)  # Proportion de poids à zéro
    p_u = sparsity
    p_s = 0.0  # Pas de structured pruning

    q_w = quantization_ratio  # Quantification des poids
    q_a = quantization_ratio  # Quantification des activations

    score = (1 - (p_s + p_u)) * (q_w / 32) * params / (5.6 * 10**6) + (1 - p_s) * (max(q_w, q_a) / 32) * ops / (2.8 * 10**8)
    return score, sparsity


def plot_pruning_sparsity_by_layer(model):
    layer_names = []
    sparsity_values = []

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nonzero = 0
            total = 0
            for param in module.parameters():
                nonzero += param.nonzero().size(0)
                total += param.numel()
            sparsity = 100 * (1 - nonzero / total) if total != 0 else 0

            layer_names.append(name)
            sparsity_values.append(sparsity)

    # Plot
    plt.figure(figsize=(14, 6))
    plt.bar(layer_names, sparsity_values)
    plt.xticks(rotation=90)
    plt.ylabel('Sparsité (%)')
    plt.xlabel('Couche')
    plt.title('Sparsité par couche après pruning')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("sparsite_par_couche.png")
    plt.show()

# Exemple d'utilisation
params = count_parameters(model)  # Calcul du nombre de paramètres du modèle
macs = count_macs(model, input_size=(3, 32, 32))  # Entrée de taille (3, 32, 32) pour CIFAR-10
score = compute_score_unstructured(model, quantization_ratio=16, params=params, ops=macs)

print(f"Nombre total de paramètres du modèle : {params}")
print_pruning_stats(model)
print(f"Nombre approximatif de MACs : {macs}")
print(f"Score du modèle : {score}")

plot_pruning_sparsity_by_layer(model)
