import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models import resnet
from torch.amp import autocast
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)

# Utility Functions
def prune_model(model, pruning_ratio, structured):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if structured:
                prune.ln_structured(module, name='weight', amount=pruning_ratio, dim=0, n=1)
            else:
                prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
            prune.remove(module, 'weight')
    return model

def binarize_model(model):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            with torch.no_grad():
                module.weight.data = module.weight.data.sign()
    return model

def evaluate_model(model, use_half):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device).long()
            if use_half:
                inputs = inputs.half()
                with autocast(device_type='cuda', dtype=torch.half):
                    outputs = model(inputs)
            else:
                outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_macs(model, input_size=(3, 32, 32), use_half=False):
    x = torch.ones(1, *input_size).to(device)
    if use_half:
        x = x.half()
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

def compute_score(pruning_ratio, q_w, params, ops):
    base_w, base_f = 5.6e6, 2.8e8
    q_a = 16
    return (1 - pruning_ratio) * (q_w / 32) * params / base_w + (1 - pruning_ratio) * (max(q_w, q_a) / 32) * ops / base_f

# Experiment Modes
modes = [
    {'name': 'half', 'half': True, 'binary': False, 'prune': False},
    {'name': 'binary', 'half': False, 'binary': True, 'prune': False},
    {'name': 'half_structured', 'half': True, 'binary': False, 'prune': True, 'structured': True},
    {'name': 'half_unstructured', 'half': True, 'binary': False, 'prune': True, 'structured': False},
    {'name': 'binary_structured', 'half': False, 'binary': True, 'prune': True, 'structured': True},
    {'name': 'binary_unstructured', 'half': False, 'binary': True, 'prune': True, 'structured': False},
]

pruning_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
results = {mode['name']: [] for mode in modes}

for mode in modes:
    print(f"\n=== {mode['name'].upper()} ===\n")
    for ratio in (pruning_rates if mode['prune'] else [0.0]):
        model = resnet.ResNet18().to(device)
        if mode['half']:
            model = model.half()
        model.load_state_dict(torch.load("pth_files/comb.pth"))
        if mode['prune']:
            model = prune_model(model, pruning_ratio=ratio, structured=mode['structured'])

        optimizer = optim.SGD(model.parameters(), lr=0.03584, weight_decay=0.00235)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.28)
        criterion = nn.CrossEntropyLoss()

        acc_epochs, loss_epochs = [], []
        for epoch in range(10):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                if mode['half']:
                    inputs = inputs.half()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                if mode['binary']:
                    binarize_model(model)
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            scheduler.step()
            acc_epochs.append(100 * correct / total)
            loss_epochs.append(running_loss / len(trainloader))

        acc = evaluate_model(model, mode['half'])
        params = count_parameters(model)
        macs = count_macs(model, use_half=mode['half'])
        q_w = 1 if mode['binary'] else (16 if mode['half'] else 32)
        score = compute_score(ratio, q_w, params, macs)

        results[mode['name']].append({
            'ratio': ratio,
            'final_acc': acc,
            'acc_epochs': acc_epochs,
            'loss_epochs': loss_epochs,
            'score': score
        })
        print(f"Ratio {ratio:.2f}: Final Accuracy {acc:.2f}%, Score {score:.4f}")

# Generate Plots
for metric, ylabel, filename in [
    ('final_acc', 'Final Accuracy (%)', 'accuracy_vs_pruning_rate.png'),
    ('score', 'Score', 'score_vs_pruning_rate.png')
]:
    plt.figure()
    for mode in results:
        plt.plot([r['ratio'] * 100 for r in results[mode]],
                 [r[metric] for r in results[mode]],
                 marker='o', label=mode)
    plt.xlabel('Pruning Rate (%)')
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} vs Pruning Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, dpi=300)
    plt.close()

# Plot Accuracy & Loss per Epoch for 50% pruning
plt.figure()
for mode in results:
    if results[mode][-1]['ratio'] == 0.5:
        r = results[mode][-1]
        plt.plot(range(1, len(r['acc_epochs']) + 1), r['acc_epochs'], label=f'{mode} Accuracy')
        plt.plot(range(1, len(r['loss_epochs']) + 1), r['loss_epochs'], label=f'{mode} Loss')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Accuracy and Loss over Epochs at 50% Pruning')
plt.legend()
plt.grid(True)
plt.savefig("plot_results/accuracy_loss_vs_epochs.png", dpi=300)
plt.close()

# Plot Final Accuracy for all modes (no pruning rate)
plt.figure()
for mode in results:
    plt.plot([r['final_acc'] for r in results[mode]], marker='o', label=mode)
plt.xlabel('Configuration Index')
plt.ylabel('Final Accuracy (%)')
plt.title('Final Accuracy per Mode')
plt.legend()
plt.grid(True)
plt.savefig("plot_results/final_accuracy_per_mode.png", dpi=300)
plt.close()

# Plot Score for all modes (no pruning rate)
plt.figure()
for mode in results:
    plt.plot([r['score'] for r in results[mode]], marker='o', label=mode)
plt.xlabel('Configuration Index')
plt.ylabel('Score')
plt.title('Score per Mode')
plt.legend()
plt.grid(True)
plt.savefig("plot_results/score_per_mode.png", dpi=300)
plt.close()

# Plot Accuracy per Epoch for all modes at their last config (e.g., 50% pruning or no pruning)
plt.figure()
for mode in results:
    r = results[mode][-1]
    plt.plot(range(1, len(r['acc_epochs']) + 1), r['acc_epochs'], label=f'{mode} Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy per Epoch for All Modes (Last Config)')
plt.legend()
plt.grid(True)
plt.savefig("plot_results/accuracy_per_epoch_all_modes.png", dpi=300)
plt.close()

# Plot Loss per Epoch for all modes at their last config
plt.figure()
for mode in results:
    r = results[mode][-1]
    plt.plot(range(1, len(r['loss_epochs']) + 1), r['loss_epochs'], label=f'{mode} Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch for All Modes (Last Config)')
plt.legend()
plt.grid(True)
plt.savefig("plot_results/loss_per_epoch_all_modes.png", dpi=300)
plt.close()
