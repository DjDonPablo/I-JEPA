import torch
import os
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.models import VisionTransformer
from src.new_dataset import CIFAR10Dataset
from tqdm import tqdm


class LinearProbe(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vit = VisionTransformer(
            image_size=32,
            patch_size=4,
            num_layers=6,
            num_heads=4,
            hidden_dim=128,
            mlp_dim=128 * 4,
            num_classes=num_classes,
        )

    def forward(self, x):
        x = self.vit(x)
        return x


def train_linear_probing(train_loader, device, val_loader):
    model = LinearProbe(10).to(device)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    model.train()
    train_loss = 0
    train_samples = 0
    train_correct = 0
    for img, label in tqdm(train_loader):
        img = img.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        prediction = model(img)
        loss = criterion(prediction, label)

        loss.backward()
        optimizer.step()

        pred = prediction.argmax(dim=1)
        train_correct += (pred == label.argmax(dim=1)).sum().item()

        train_loss += loss.item()
        train_samples += len(img)

    model.eval()
    eval_loss = 0
    eval_samples = 0
    eval_accuracy = 0
    for data, label in val_loader:
        data, label = data.to(device), label.to(device)

        preds = model(data)
        loss = criterion(preds, label)

        pred = preds.argmax(dim=1)
        eval_accuracy += (pred == label.argmax(dim=1)).sum().item()

        eval_loss += loss.item()
        eval_samples += len(data)

    train_accuracy = train_correct / train_samples
    eval_accuracy /= eval_samples
    return (
        train_loss / train_samples,
        train_accuracy,
        eval_loss / eval_samples,
        eval_accuracy,
    )


if __name__ == "__main__":
    epochs = 100
    batch_size = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = os.path.join("..", "cifar-10")
    train_dataset = CIFAR10Dataset(data_path, "supervised", "train")
    test_dataset = CIFAR10Dataset(data_path, "supervised", "test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        train_loss, train_acc, val_loss, val_acc = train_linear_probing(
            train_loader=train_loader, device=device, val_loader=val_loader
        )

        print(
            f"Epoch [{epoch + 1}/{epochs}], train_loss : {train_loss}, train_acc : {train_acc}, val_loss : {val_loss}, val_acc : {val_acc}"
        )
