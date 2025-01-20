import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.new_dataset import STL10DatasetLabelled

from src.model.ijepa import IJEPA
from tqdm import tqdm


class LinearProbe(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.linear = nn.Linear(384, num_classes)
        self.ijepa = IJEPA(
            evaluation_on=True,
            nb_mask=4,
            image_size=96,
            patch_size=12,
            embed_dim=384,
            num_heads=8,
            num_layers=8
        )

        checkpoint = torch.load('checkpoint_new_ds.pth')
        self.ijepa.load_state_dict(checkpoint["model"]) # TODO: load state_dict
        for parameters in self.ijepa.parameters():
            parameters.requires_grad = False


    def forward(self, x):
        with torch.no_grad():
            # Get embeddings from context encoder without masks
            x = self.ijepa.context_encoder(x)  # B, N+1, D

        # Use CLS token or mean of patch embeddings
        x = x[:, 0]  # Use CLS token (B, D)
        # Or use mean: x = x[:, 1:].mean(dim=1)  # Skip CLS token, mean over patches

        x = self.linear(x)  # B, num_classes
        return x

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            pred = outputs.argmax(dim=1)
            if len(labels.shape) > 1:  # If labels are one-hot encoded
                labels = labels.argmax(dim=1)
            correct = (pred == labels).sum().item()

            total_loss += loss.item()
            total_correct += correct
            total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


def train_linear_probing(train_loader, val_loader, device, num_epochs=30):
    # linear evaluation training
    model = LinearProbe(8).to(device)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_acc = 0
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_samples = 0

        for img, label in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            img = img.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            prediction = model(img)

            if len(label.shape) > 1:  # If labels are one-hot encoded
                label = label.argmax(dim=1)

            loss = criterion(prediction, label)
            loss.backward()
            optimizer.step()

            pred = prediction.argmax(dim=1)
            correct = (pred == label).sum().item()

            train_loss += loss.item()
            train_correct += correct
            train_samples += label.size(0)

        # Calculate training metrics
        train_loss = train_loss / train_samples
        train_acc = train_correct / train_samples

        # Evaluate on validation set
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
            }, 'best_linear_probe.pth')

        scheduler.step()

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print("------------------------")

    return best_model


if __name__ == "__main__":
    # Training Hyp. Param.
    epochs = 30
    batch_size = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = STL10DatasetLabelled(
        path_images="src/stl10_binary/train_X.bin",  # TODO : adapt to your path
        path_labels="src/stl10_binary/train_y.bin",
    )

    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset, val_dataset = random_split(dataset, [0.8, 0.1, 0.1], generator=generator)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True
    )

    best_model = train_linear_probing(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=epochs
    )

    model = LinearProbe(8).to(device)
    model.load_state_dict(best_model)
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal Test Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
