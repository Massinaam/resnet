# import torch
# import torch.nn as nn
# import torch.nn.utils.prune as prune
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from models import resnet
# from torch.amp import autocast
# import torch.optim as optim
# import matplotlib.pyplot as plt
# import numpy as np
# import os

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Data
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)

# # Utility Functions
# def prune_model(model, pruning_ratio, structured):
#     for module in model.modules():
#         if isinstance(module, (nn.Conv2d, nn.Linear)):
#             if structured:
#                 prune.ln_structured(module, name='weight', amount=pruning_ratio, dim=0, n=1)
#             else:
#                 prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
#             prune.remove(module, 'weight')
#     return model

# def evaluate_model(model):
#     model.eval()
#     correct, total = 0, 0
#     with torch.no_grad():
#         for inputs, labels in testloader:
#             inputs, labels = inputs.to(device), labels.to(device).long()
#             inputs = inputs.half()
#             with autocast(device_type='cuda', dtype=torch.half):
#                 outputs = model(inputs)
#             _, predicted = outputs.max(1)
#             correct += predicted.eq(labels).sum().item()
#             total += labels.size(0)
#     return 100. * correct / total

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# def count_macs(model, input_size=(3, 32, 32)):
#     x = torch.ones(1, *input_size).half().to(device)
#     macs = 0
#     def count_conv_macs(module, inp, out):
#         nonlocal macs
#         if isinstance(module, nn.Conv2d):
#             macs += np.prod(out.shape[2:]) * np.prod(module.weight.shape[1:])
#     def count_linear_macs(module, inp, out):
#         nonlocal macs
#         if isinstance(module, nn.Linear):
#             macs += inp[0].shape[1] * out.shape[1]
#     hooks = []
#     for m in model.modules():
#         if isinstance(m, nn.Conv2d):
#             hooks.append(m.register_forward_hook(count_conv_macs))
#         elif isinstance(m, nn.Linear):
#             hooks.append(m.register_forward_hook(count_linear_macs))
#     model(x)
#     for h in hooks:
#         h.remove()
#     return macs

# def compute_score(pruning_ratio, q_w, params, ops):
#     base_w, base_f = 5.6e6, 2.8e8
#     q_a = 16
#     return (1 - pruning_ratio) * (q_w / 32) * params / base_w + (1 - pruning_ratio) * (max(q_w, q_a) / 32) * ops / base_f

# # Experiment Parameters
# pruning_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
# results = {'structured': [], 'unstructured': []}

# for mode in ['structured', 'unstructured']:
#     print(f"\n=== {mode.upper()} PRUNING ===\n")
#     for ratio in pruning_rates:
#         model = resnet.ResNet18().to(device).half()
#         model.load_state_dict(torch.load("pth_files/comb.pth"))
#         model = prune_model(model, pruning_ratio=ratio, structured=(mode == 'structured'))

#         optimizer = optim.SGD(model.parameters(), lr=0.03584, weight_decay=0.00235)
#         scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.28)
#         criterion = nn.CrossEntropyLoss()

#         acc_epochs, loss_epochs = [], []
#         for epoch in range(10):
#             model.train()
#             running_loss, correct, total = 0.0, 0, 0
#             for inputs, labels in trainloader:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 inputs = inputs.half()
#                 optimizer.zero_grad()
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 optimizer.step()
#                 running_loss += loss.item()
#                 _, predicted = outputs.max(1)
#                 total += labels.size(0)
#                 correct += predicted.eq(labels).sum().item()
#             scheduler.step()
#             acc_epochs.append(100 * correct / total)
#             loss_epochs.append(running_loss / len(trainloader))

#         acc = evaluate_model(model)
#         params = count_parameters(model)
#         macs = count_macs(model)
#         score = compute_score(ratio, 16, params, macs)

#         results[mode].append({
#             'ratio': ratio,
#             'final_acc': acc,
#             'acc_epochs': acc_epochs,
#             'loss_epochs': loss_epochs,
#             'score': score
#         })
#         print(f"Ratio {ratio:.2f}: Final Accuracy {acc:.2f}%, Score {score:.4f}")

# # Plot Accuracy vs Pruning Rate
# plt.figure()
# for mode in ['structured', 'unstructured']:
#     plt.plot([r['ratio'] * 100 for r in results[mode]],
#              [r['final_acc'] for r in results[mode]],
#              label=mode)
# plt.xlabel('Pruning Rate (%)')
# plt.ylabel('Final Accuracy (%)')
# plt.title('Final Accuracy vs Pruning Rate')
# plt.legend()
# plt.grid(True)
# plt.savefig("accuracy_vs_pruning_rate.png", dpi=300)
# plt.close()

# # Plot Accuracy & Loss per Epoch for last pruning level (0.5)
# plt.figure()
# for mode in ['structured', 'unstructured']:
#     r = results[mode][-1]
#     plt.plot(range(1, len(r['acc_epochs']) + 1), r['acc_epochs'], label=f'{mode} Accuracy')
#     plt.plot(range(1, len(r['loss_epochs']) + 1), r['loss_epochs'], label=f'{mode} Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Value')
# plt.title('Accuracy and Loss over Epochs at 50% Pruning')
# plt.legend()
# plt.grid(True)
# plt.savefig("accuracy_loss_vs_epochs.png", dpi=300)
# plt.close()

# # Plot Pruning Rate / Accuracy Tradeoff
# plt.figure()
# for mode in ['structured', 'unstructured']:
#     plt.plot([r['ratio'] * 100 for r in results[mode]],
#              [r['final_acc'] for r in results[mode]],
#              marker='o', label=mode)
# plt.xlabel('Pruning Rate (%)')
# plt.ylabel('Final Accuracy (%)')
# plt.title('Pruning Rate / Accuracy Tradeoff')
# plt.legend()
# plt.grid(True)
# plt.savefig("pruning_rate_accuracy_tradeoff.png", dpi=300)
# plt.close()

# # Plot Score vs Pruning Rate
# plt.figure()
# for mode in ['structured', 'unstructured']:
#     plt.plot([r['ratio'] * 100 for r in results[mode]],
#              [r['score'] for r in results[mode]],
#              marker='o', label=mode)
# plt.xlabel('Pruning Rate (%)')
# plt.ylabel('Score')
# plt.title('Score vs Pruning Rate')
# plt.legend()
# plt.grid(True)
# plt.savefig("score_vs_pruning_rate.png", dpi=300)
# plt.close()
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms
from models import resnet
from torch.amp import autocast
import matplotlib.pyplot as plt
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chargement des données uniquement pour l’évaluation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)

# Fonctions utilitaires
def prune_model(model, pruning_ratio, structured):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if structured:
                prune.ln_structured(module, name='weight', amount=pruning_ratio, dim=0, n=1)
            else:
                prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
            prune.remove(module, 'weight')
    return model

def evaluate_model(model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device).long()
            inputs = inputs.half()
            with autocast(device_type='cuda', dtype=torch.half):
                outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_macs(model, input_size=(3, 32, 32)):
    x = torch.ones(1, *input_size).half().to(device)
    macs = 0
    def count_conv_macs(module, inp, out):
        nonlocal macs
        if isinstance(module, nn.Conv2d):
            macs += np.prod(out.shape[2:]) * np.prod(module.weight.shape[1:])
    def count_linear_macs(module, inp, out):
        nonlocal macs
        if isinstance(module, nn.Linear):
            macs += inp[0].shape[1] * out.shape[1]
    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(count_conv_macs))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(count_linear_macs))
    model(x)
    for h in hooks:
        h.remove()
    return macs

def compute_sparsity(model):
    total_zeros, total_params = 0, 0
    for param in model.parameters():
        total_zeros += torch.sum(param == 0).item()
        total_params += param.numel()
    return total_zeros / total_params

def compute_score(pruning_ratio, q_w, params, ops):
    base_w, base_f = 5.6e6, 2.8e8
    q_a = 16
    return (1 - pruning_ratio) * (q_w / 32) * params / base_w + (1 - pruning_ratio) * (max(q_w, q_a) / 32) * ops / base_f

# Taux de pruning
pruning_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
results = {'structured': [], 'unstructured': []}

# Expériences sans entraînement
for mode in ['structured', 'unstructured']:
    for ratio in pruning_rates:
        model = resnet.ResNet18().to(device).half()
        model.load_state_dict(torch.load("pth_files/comb.pth"))
        model = prune_model(model, pruning_ratio=ratio, structured=(mode == 'structured'))

        acc = evaluate_model(model)
        params = count_parameters(model)
        macs = count_macs(model)
        sparsity = compute_sparsity(model)
        score = compute_score(ratio, 16, params, macs)

        results[mode].append({
            'ratio': ratio,
            'final_acc': acc,
            'sparsity': sparsity * 100,
            'params': params,
            'macs': macs,
            'score': score
        })
        print(f"{mode.upper()} | Ratio {ratio:.1f} | Acc {acc:.2f}% | Sparsity {sparsity*100:.1f}% | Score {score:.4f}")

# === Visualisations ===

# Accuracy vs Pruning Rate
plt.figure()
for mode in ['structured', 'unstructured']:
    plt.plot([r['ratio'] * 100 for r in results[mode]],
             [r['final_acc'] for r in results[mode]], marker='o', label=mode)
plt.xlabel('Pruning Rate (%)')
plt.ylabel('Final Accuracy (%)')
plt.title('Final Accuracy vs Pruning Rate')
plt.legend()
plt.grid(True)
plt.savefig("accuracy_vs_pruning_rate.png", dpi=300)
plt.close()

# Sparsity vs Pruning Rate
plt.figure()
for mode in ['structured', 'unstructured']:
    plt.plot([r['ratio'] * 100 for r in results[mode]],
             [r['sparsity'] for r in results[mode]], marker='o', label=mode)
plt.xlabel('Pruning Rate (%)')
plt.ylabel('Sparsity (%)')
plt.title('Sparsity vs Pruning Rate')
plt.legend()
plt.grid(True)
plt.savefig("sparsity_vs_pruning_rate.png", dpi=300)
plt.close()

# Accuracy vs Sparsity
plt.figure()
for mode in ['structured', 'unstructured']:
    plt.plot([r['sparsity'] for r in results[mode]],
             [r['final_acc'] for r in results[mode]], marker='o', label=mode)
plt.xlabel('Sparsity (%)')
plt.ylabel('Final Accuracy (%)')
plt.title('Final Accuracy vs Sparsity')
plt.legend()
plt.grid(True)
plt.savefig("accuracy_vs_sparsity.png", dpi=300)
plt.close()

# Score vs Pruning Rate
plt.figure()
for mode in ['structured', 'unstructured']:
    plt.plot([r['ratio'] * 100 for r in results[mode]],
             [r['score'] for r in results[mode]], marker='o', label=mode)
plt.xlabel('Pruning Rate (%)')
plt.ylabel('Score')
plt.title('Score vs Pruning Rate')
plt.legend()
plt.grid(True)
plt.savefig("score_vs_pruning_rate.png", dpi=300)
plt.close()

# Sauvegarde facultative des résultats
import pickle
with open('results_no_train.pkl', 'wb') as f:
    pickle.dump(results, f)
