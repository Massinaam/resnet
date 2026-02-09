import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt
import os
from models import resnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path="comb.pth"):
    model = resnet.ResNet18()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.half().to(device)
    return model

def prune_unstructured(model, pruning_ratio):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
            prune.remove(module, 'weight')
    return model

def make_dense(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, 'weight_mask'):
            with torch.no_grad():
                module.weight.data.mul_(module.weight_mask)
            prune.remove(module, 'weight')
    return model

def save_model(model, output_path):
    torch.save(model.state_dict(), output_path)
    print(f"✅ Modèle sauvegardé : {output_path}")

def get_model_size(model):
    torch.save(model.state_dict(), "temp.pth")
    size_mb = os.path.getsize("temp.pth") / (1024 * 1024)
    os.remove("temp.pth")
    return size_mb

def compute_sparsity(model):
    total = sum(p.numel() for p in model.parameters())
    nonzero = sum(p.nonzero().size(0) for p in model.parameters())
    sparsity = 100 * (1 - nonzero / total)
    return total, nonzero, sparsity

# === MAIN ===
if __name__ == "__main__":
    ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    param_counts = []
    sparsities = []
    model_sizes = []

    for ratio in ratios:
        print(f"\n🔧 Pruning non structuré avec ratio {ratio}")
        model = load_model("comb.pth")
        model = prune_unstructured(model, pruning_ratio=ratio)
        model = make_dense(model)

        output_filename = f"resnet18_pruned_unstructured_{int(ratio*100)}.pth"
        save_model(model, output_filename)

        total, nonzero, sparsity = compute_sparsity(model)
        size = get_model_size(model)

        param_counts.append(nonzero)
        sparsities.append(sparsity)
        model_sizes.append(size)

    # === PLOT ===
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(ratios, param_counts, marker='o', label="Paramètres non nuls")
    plt.plot(ratios, model_sizes, marker='s', label="Taille (Mo)")
    plt.xlabel("Taux de pruning non structuré")
    plt.ylabel("Valeur")
    plt.title("Évolution des paramètres et taille modèle")
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(ratios, sparsities, marker='^', label="Sparsité (%)")
    plt.xlabel("Taux de pruning non structuré")
    plt.ylabel("Sparsité")
    plt.title("Sparsité du modèle")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig("pruning_non_structured_analysis.png")
    plt.show()
