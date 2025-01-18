import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from src.our_code import ViTEncoder
from src.dataset.dataset import JEPADataset
from src.mask.mask import generate_masks


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Training Loop
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = (
            images.to(device),
            labels.to(device),
        )
        context_mask = generate_masks(4, 8, 32)

        # Forward pass
        outputs = model(images, context_mask)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


# Evaluation Loop
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = (
                images.to(device),
                labels.to(device),
            )

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


if __name__ == "__main__":
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-3

    dataset = JEPADataset(
        dataset_path="dataset/archive",
        labels_filename="labels.csv",
    )

    generator1 = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(
        dataset, [0.8, 0.2], generator=generator1
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = ViTEncoder(num_classes=8)
    model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Training and Evaluation
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
