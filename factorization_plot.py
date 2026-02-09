import pickle
import matplotlib.pyplot as plt
import os

# Fonction de chargement des métriques
def load_metrics(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

# Liste des modèles à comparer
model_files = {
    "Standard ResNet18": "pkl_files/resnet18_standard_metrics.pkl",
    "Factorized ResNet18": "pkl_files/resnet18_factorized_metrics.pkl"
}

# Préparation des figures
plt.figure(figsize=(12, 5))

# Sous-figure 1 : Loss
plt.subplot(1, 2, 1)
for label, file in model_files.items():
    if os.path.exists(file):
        metrics = load_metrics(file)
        epochs = list(range(1, len(metrics["train_loss"]) + 1))
        plt.plot(epochs, metrics["train_loss"], label=f"{label} Train Loss")
        if "test_loss" in metrics:
            plt.plot(epochs, metrics["test_loss"], linestyle='--', label=f"{label} Test Loss")
plt.title("Évolution de la perte")
plt.xlabel("Époque")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

# Sous-figure 2 : Accuracy
plt.subplot(1, 2, 2)
for label, file in model_files.items():
    if os.path.exists(file):
        metrics = load_metrics(file)
        epochs = list(range(1, len(metrics["train_acc"]) + 1))
        plt.plot(epochs, metrics["train_acc"], label=f"{label} Train Acc")
        plt.plot(epochs, metrics["test_acc"], linestyle='--', label=f"{label} Test Acc")
plt.title("Évolution de l’accuracy")
plt.xlabel("Époque")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.legend()

# Sauvegarde du graphique
os.makedirs("plot_results", exist_ok=True)
plt.tight_layout()
plt.savefig("plot_results/comparative_plot.png")
plt.show()
