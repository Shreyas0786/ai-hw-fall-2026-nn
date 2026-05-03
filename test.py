import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.mlp import MLP
from models.cnn import CNN
from models.transformer import TransformerEncoder
import os

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

def test_model(model, name):
    model.load_state_dict(torch.load(f"saved_models/{name}.pth", map_location=device))
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"{name.upper()} Accuracy: {accuracy:.2f}%")
    return accuracy

results = {}
results["MLP"] = test_model(MLP(), "mlp")
results["CNN"] = test_model(CNN(), "cnn")
results["Transformer"] = test_model(TransformerEncoder(), "transformer")

os.makedirs("results", exist_ok=True)
with open("results/results.txt", "w") as f:
    for model_name, acc in results.items():
        f.write(f"{model_name}: {acc:.2f}%\n")

print("\nResults saved to results/results.txt")