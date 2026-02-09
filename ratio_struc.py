import torch
import torch.nn as nn
import torch_pruning as tp
import matplotlib.pyplot as plt
import os
from models import resnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = resnet.ResNet18()
    model.load_state_dict(torch.load("comb.pth", map_location=device))
    model = model.to(device).eval()
    return model

def get_model_size(model, temp_path="temp.pth"):
    torch.save(model.state_dict(), temp_path)
    size_mb = os.path.getsize(temp_path) 
    os.remove(temp_path)
    return size_mb

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def apply_structured_real_pruning(model, pruning_ratio):
    example_inputs = torch.randn(1, 3, 32, 32).to(device)
    DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)

    def prune_conv_layer(layer, ratio):
        strategy = tp.strategy.L1Strategy()
        pruning_idxs = strategy(layer.weight, amount=ratio)
        plan = DG.get_pruning_plan(layer, tp.prune_conv_out_channel, pruning_idxs)
        plan.exec()

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            try:
                prune_conv_layer(m, pruning_ratio)
            except Exception as e:
                print(f"⚠️ Skipped layer {m}: {e}")
    return model

# === Analyse pour différents taux ===
ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
param_counts = []
model_sizes = []

for ratio in ratios:
    print(f"\n🔧 Pruning structuré réel à {int(ratio * 100)}%...")
    model = load_model()
    if ratio > 0.0:
        model = apply_structured_real_pruning(model, ratio)
    params = count_parameters(model)
    size = get_model_size(model)
    param_counts.append(params)
    model_sizes.append(size)

    out_path = f"resnet18_structured_real_{int(ratio*100)}.pth"
    torch.save(model, f"resnet18_pruned_structured_real_{int(ratio*100)}.pth")
    print(f"📦 Modèle sauvegardé : {out_path} — {params:,} params, {size:.2f} Mo")

model = torch.load("resnet18_pruned_structured_real_40.pth")
model.eval()

# === Plot ===
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(ratios, param_counts, marker='o')
plt.xlabel("Taux de pruning structuré (%)")
plt.ylabel("Nombre de paramètres restants")
plt.title("Paramètres après pruning structuré réel")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(ratios, model_sizes, marker='s')
plt.xlabel("Taux de pruning structuré (%)")
plt.ylabel("Taille du modèle (Mo)")
plt.title("Taille du modèle après pruning structuré réel")
plt.grid(True)

plt.tight_layout()
plt.savefig("structured_real_pruning_plot.png")
plt.show()
