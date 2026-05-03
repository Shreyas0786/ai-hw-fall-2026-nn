import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.mlp import MLP
from models.cnn import CNN
from models.transformer import TransformerEncoder
import os

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("saved_models", exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

def train_model(model, name, epochs=5):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print(f"\nTraining {name}...")
    loss_history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_loss = total_loss/len(train_loader)
        loss_history.append(epoch_loss)
        print(f"  Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), f"saved_models/{name}.pth")
    print(f"  Saved {name}.pth")
    return loss_history

mlp_loss = train_model(MLP(), "mlp")
cnn_loss = train_model(CNN(), "cnn")
transformer_loss = train_model(TransformerEncoder(), "transformer")

import json
os.makedirs("results", exist_ok=True)
with open("results/loss_history.json", "w") as f:
    json.dump({"mlp": mlp_loss, "cnn": cnn_loss, "transformer": transformer_loss}, f)