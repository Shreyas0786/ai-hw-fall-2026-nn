import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.mlp import MLP
from models.cnn import CNN
from models.transformer import TransformerEncoder
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("results", exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_data = datasets.MNIST(root="data", train=False, download=False, transform=transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# --- Load accuracies from results.txt ---
accuracies = {}
with open("results/results.txt", "r") as f:
    for line in f:
        name, acc = line.strip().split(": ")
        accuracies[name] = float(acc.replace("%", ""))

# --- Load loss history ---
with open("results/loss_history.json", "r") as f:
    loss_history = json.load(f)

# --- Get predictions for confusion matrix ---
def get_predictions(model, name):
    model.load_state_dict(torch.load(f"saved_models/{name}.pth", map_location=device))
    model = model.to(device)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)

mlp_preds, labels = get_predictions(MLP(), "mlp")
cnn_preds, _ = get_predictions(CNN(), "cnn")
transformer_preds, _ = get_predictions(TransformerEncoder(), "transformer")

# --- Plot 1: Bar Chart ---
plt.figure(figsize=(8, 5))
colors = ["#4C72B0", "#DD8452", "#55A868"]
bars = plt.bar(accuracies.keys(), accuracies.values(), color=colors, width=0.5)
plt.ylim(94, 100)
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison")
for bar, val in zip(bars, accuracies.values()):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f"{val:.2f}%", ha="center", va="bottom", fontweight="bold")
plt.tight_layout()
plt.savefig("results/bar_chart.png", dpi=150)
plt.close()
print("Saved bar_chart.png")

# --- Plot 2: Loss Curves ---
plt.figure(figsize=(8, 5))
epochs = range(1, 6)
plt.plot(epochs, loss_history["mlp"], marker="o", label="MLP", color="#4C72B0")
plt.plot(epochs, loss_history["cnn"], marker="o", label="CNN", color="#DD8452")
plt.plot(epochs, loss_history["transformer"], marker="o", label="Transformer", color="#55A868")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curves")
plt.legend()
plt.tight_layout()
plt.savefig("results/loss_curves.png", dpi=150)
plt.close()
print("Saved loss_curves.png")

# --- Plot 3: Confusion Matrices ---
def plot_confusion_matrix(preds, labels, title, filename):
    matrix = np.zeros((10, 10), dtype=int)
    for p, l in zip(preds, labels):
        matrix[l][p] += 1
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(range(10))
    plt.yticks(range(10))
    for i in range(10):
        for j in range(10):
            plt.text(j, i, str(matrix[i][j]), ha="center", va="center",
                     color="white" if matrix[i][j] > 500 else "black", fontsize=7)
    plt.tight_layout()
    plt.savefig(f"results/{filename}", dpi=150)
    plt.close()
    print(f"Saved {filename}")

plot_confusion_matrix(mlp_preds, labels, "Confusion Matrix - MLP", "confusion_mlp.png")
plot_confusion_matrix(cnn_preds, labels, "Confusion Matrix - CNN", "confusion_cnn.png")
plot_confusion_matrix(transformer_preds, labels, "Confusion Matrix - Transformer", "confusion_transformer.png")

print("\nAll visualizations saved to results/")