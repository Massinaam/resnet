# BinaryConnect Implementation with Visualizations and Model Size/Throughput Analysis
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy
import pickle
import matplotlib.pyplot as plt
from models import resnet

# BinaryConnect Definition
class BC():
    def __init__(self, model):
        count_targets = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear))
        self.bin_range = numpy.linspace(0, count_targets - 1, count_targets).astype('int').tolist()
        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.target_modules = []
        self.model = model
        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                index += 1
                if index in self.bin_range:
                    self.saved_params.append(m.weight.data.clone())
                    self.target_modules.append(m.weight)

    def save_params(self):
        for idx in range(self.num_of_params):
            self.saved_params[idx].copy_(self.target_modules[idx].data)

    def binarization(self):
        self.save_params()
        for idx in range(self.num_of_params):
            self.target_modules[idx].data.copy_(self.target_modules[idx].data.sign())

    def restore(self):
        for idx in range(self.num_of_params):
            self.target_modules[idx].data.copy_(self.saved_params[idx])

    def clip(self):
        hardtanh = nn.Hardtanh(min_val=-1.0, max_val=1.0)
        for idx in range(self.num_of_params):
            self.target_modules[idx].data.copy_(hardtanh(self.target_modules[idx].data))

    def forward(self, x):
        return self.model(x)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
lr = 0.0358408115435323
optimizer = 'AdamW'
weight_decay = 0.002345625688196673
step_size = 50
gamma = 0.28261595872305967
n_epochs = 70

# Data Loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

# Model Initialization
mymodel = resnet.ResNet18()
mymodelbc = BC(mymodel)
mymodelbc.model = mymodelbc.model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(mymodelbc.model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# Training Loop
losses, accuracies, test_losses, test_accuracies = [], [], [], []

print('🚀 Start Training...')
for epoch in range(n_epochs):
    mymodelbc.model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        mymodelbc.binarization()
        optimizer.zero_grad()
        outputs = mymodelbc.model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        mymodelbc.restore()
        optimizer.step()
        mymodelbc.clip()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    scheduler.step()
    epoch_accuracy = 100. * correct / total
    epoch_loss = running_loss / len(trainloader)
    accuracies.append(epoch_accuracy)
    losses.append(epoch_loss)

    print(f"✅ Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_accuracy:.2f}%")

    # Evaluation
    mymodelbc.model.eval()
    mymodelbc.binarization()
    epoch_test_loss, epoch_test_correct, epoch_test_total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = mymodelbc.model(images)
            loss = criterion(outputs, labels)
            epoch_test_loss += loss.item()
            _, predicted = outputs.max(1)
            epoch_test_total += labels.size(0)
            epoch_test_correct += predicted.eq(labels).sum().item()

    mymodelbc.restore()
    test_loss_mean = epoch_test_loss / len(testloader)
    test_acc_mean = 100. * epoch_test_correct / epoch_test_total
    test_losses.append(test_loss_mean)
    test_accuracies.append(test_acc_mean)
    print(f"🔍 Epoch {epoch+1}: Test Loss = {test_loss_mean:.4f}, Test Accuracy = {test_acc_mean:.2f}%")

# Final Test Evaluation
mymodelbc.model.eval()
mymodelbc.binarization()
test_running_loss, test_correct, test_total = 0.0, 0, 0
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

final_test_accuracy = 100. * test_correct / test_total
final_test_loss = test_running_loss / len(testloader)

# Saving Model and Metrics
os.makedirs('pkl_files', exist_ok=True)
torch.save(mymodel.state_dict(), "bc.pth")
with open('pkl_files/losses_accuracies_quantization.pkl', 'wb') as f:
    pickle.dump({'train': {'losses': losses, 'accuracies': accuracies},
                 'test': {'losses': [final_test_loss], 'accuracies': [final_test_accuracy]}}, f)

print(f"🎯 Final Test Accuracy: {final_test_accuracy:.2f}%")
print(f"🎯 Final Test Loss: {final_test_loss:.4f}")
print("Données sauvegardées dans 'pkl_files/losses_accuracies_quantization.pkl'.")

# Visualizations
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, n_epochs+1), accuracies, label='Train Accuracy')
plt.plot(range(1, n_epochs+1), test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy per Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, n_epochs+1), losses, label='Train Loss')
plt.plot(range(1, n_epochs+1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.legend()

plt.tight_layout()
plt.savefig('plot_results/accuracy_loss_per_epoch.png')
plt.show()

# Model Size Estimation
model_size_bytes = os.path.getsize("bc.pth")
model_size_mb = model_size_bytes / (1024 * 1024)
print(f"📦 Model Size: {model_size_mb:.2f} MB")

# Throughput Estimation Example (Fake Codeword Size for Illustration)
# Example: 1 codeword = 32 bits (float32) or 1 bit (binary)
bits_per_weight = 1  # BinaryConnect
num_weights = sum(p.numel() for p in mymodel.parameters())
total_bits = num_weights * bits_per_weight
throughput_mbps = total_bits / (1024 * 1024)

plt.figure()
plt.scatter([model_size_mb], [throughput_mbps])
plt.xlabel('Model Size (MB)')
plt.ylabel('Throughput (Mb)')
plt.title('Throughput vs Model Size')
plt.grid(True)
plt.savefig('plot_results/throughput_vs_model_size.png')
plt.show()
