import pickle
import matplotlib.pyplot as plt
import os

# Chargement des métriques
with open("pkl_files/resnet18_factorized_metrics.pkl", "rb") as f:
    metrics = pickle.load(f)

# Simulation nombre de paramètres si tu connais la valeur exacte (à adapter sinon)
# Exemple : 3.5M paramètres => 3_500_000
nb_params = 3500000  # adapte cette valeur si tu la connais
memory_bytes = nb_params * 4  # float32
memory_mb = memory_bytes / (1024 ** 2)

# Création du dossier pour sauvegarde
os.makedirs("figures", exist_ok=True)

# Courbes
epochs = range(1, len(metrics["train_acc"]) + 1)

plt.figure()
plt.plot(epochs, metrics["train_acc"], label="Train Accuracy")
plt.plot(epochs, metrics["test_acc"], label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy over Epochs")
plt.legend()
plt.grid()
plt.savefig("figures/accuracy_over_epochs.png")
plt.close()

plt.figure()
plt.plot(epochs, metrics["train_loss"], label="Train Loss")
plt.plot(epochs, metrics["test_loss"], label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()
plt.grid()
plt.savefig("figures/loss_over_epochs.png")
plt.close()

# Graphe final
final_test_acc = metrics["test_acc"][-1]
fig, ax1 = plt.subplots()

ax1.bar(["ResNet18 Factorized"], [final_test_acc], label="Test Accuracy (%)")
ax1.set_ylabel("Test Accuracy (%)", color="blue")

ax2 = ax1.twinx()
ax2.bar(["ResNet18 Factorized"], [memory_mb], color="orange", alpha=0.5, label="Memory (MB)")
ax2.set_ylabel("Memory Footprint (MB)", color="orange")

plt.title(f"Final Accuracy: {final_test_acc:.2f}%, Memory: {memory_mb:.2f} MB")
plt.grid()
plt.savefig("figures/final_accuracy_vs_memory.png")
plt.close()
