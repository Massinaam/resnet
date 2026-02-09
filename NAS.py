import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import pickle
import os

# Préparation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18().to(device)
model.eval()

batch_sizes = [1, 8, 16, 32, 64]
repeats = 100

results = {
    'batch_sizes': [],
    'latency_per_batch_ms': [],
    'latency_per_image_ms': [],
    'accuracy': [],
    'memory_MB': [],
    'throughput_Mbps': []
}

print("Mesure de latence, footprint mémoire et débit :\n")

for batch_size in batch_sizes:
    dummy_input = torch.randn(batch_size, 3, 32, 32).to(device)

    # Warm-up
    for _ in range(10):
        _ = model(dummy_input)

    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []

    torch.cuda.reset_peak_memory_stats()

    for _ in range(repeats):
        torch.cuda.synchronize()
        starter.record()
        _ = model(dummy_input)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings.append(curr_time)

    avg_batch_time = sum(timings) / len(timings)  # ms
    avg_time_per_image = avg_batch_time / batch_size  # ms
    memory_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB
    bits = dummy_input.numel() * 32
    throughput = (bits / avg_batch_time) / 1000  # Mbit/s

    print(f"Batch {batch_size}: {avg_batch_time:.2f} ms/batch, {avg_time_per_image:.2f} ms/image, "
          f"{memory_allocated:.2f} MB, {throughput:.2f} Mbit/s")

    results['batch_sizes'].append(batch_size)
    results['latency_per_batch_ms'].append(avg_batch_time)
    results['latency_per_image_ms'].append(avg_time_per_image)
    results['memory_MB'].append(memory_allocated)
    results['throughput_Mbps'].append(throughput)
    results['accuracy'].append(90 - 0.1 * batch_size)  # remplacer si valeurs réelles disponibles

# Sauvegarde
os.makedirs("pkl_files", exist_ok=True)
with open("pkl_files/latency_accuracy_results.pkl", "wb") as f:
    pickle.dump(results, f)

# Création des plots
os.makedirs("plot_results", exist_ok=True)

def save_plot(x, y, xlabel, ylabel, title, filename):
    plt.figure()
    plt.plot(x, y, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(f'plot_results/{filename}', dpi=300)
    plt.close()

# Courbes classiques
save_plot(results['batch_sizes'], results['latency_per_batch_ms'], "Batch size", "Latence (ms)", "Latence par batch", "latency_batch.png")
save_plot(results['batch_sizes'], results['latency_per_image_ms'], "Batch size", "Latence (ms)", "Latence par image", "latency_image.png")
save_plot(results['batch_sizes'], results['accuracy'], "Batch size", "Accuracy (%)", "Accuracy simulée", "accuracy.png")
save_plot(results['batch_sizes'], results['memory_MB'], "Batch size", "Mémoire (MB)", "Footprint mémoire", "memory.png")
save_plot(results['batch_sizes'], results['throughput_Mbps'], "Batch size", "Débit (Mbit/s)", "Throughput", "throughput.png")

# Courbes croisées
save_plot(results['latency_per_image_ms'], results['throughput_Mbps'], "Latence/image (ms)", "Débit (Mbit/s)", "Latence vs Throughput", "latency_vs_throughput.png")
save_plot(results['latency_per_image_ms'], results['accuracy'], "Latence/image (ms)", "Accuracy (%)", "Latence vs Accuracy", "latency_vs_accuracy.png")
save_plot(results['memory_MB'], results['accuracy'], "Mémoire (MB)", "Accuracy (%)", "Mémoire vs Accuracy", "memory_vs_accuracy.png")

print("\n✅ Fichiers sauvegardés :")
print(" - Résultats : pkl_files/latency_accuracy_results.pkl")
print(" - Graphiques : plot_results/*.png")