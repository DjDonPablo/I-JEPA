import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.dataset import JEPADataset

from src.model.ijepa import IJEPA
from tqdm import tqdm


class LinearProbe(nn.Module):
    def __init__(self, num_classes, embedding_dim, ijepa: IJEPA):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, num_classes)
        self.ijepa = ijepa
        self.ijepa.evaluation_on = True

        # checkpoint = torch.load("checkpoint.pth")
        # self.ijepa.load_state_dict(checkpoint["model"])  # TODO: load state_dict

    def forward(self, x):
        with torch.no_grad():
            x = self.ijepa(x)  # B, N, D
        x = x.mean(dim=1)  # B, D
        x = self.linear(x)  # B, 8
        return x


def train_linear_probing(loader, device):
    # linear evaluation training
    model = LinearProbe(8).to(device)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    train_loss = 0
    train_samples = 0
    train_correct = 0
    for img, label in loader:
        img = img.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        # infer linear probe
        prediction = model(img)
        label = label.argmax(dim=1)
        loss = criterion(prediction, label)

        loss.backward()
        optimizer.step()

        pred = prediction.argmax(dim=1)
        train_correct += (pred == label).sum().item()

        train_loss += loss.item()
        train_samples += len(img)

    accuracy = train_correct / train_samples
    return accuracy, train_loss, train_samples


if __name__ == "__main__":
    # Training Hyp. Param.
    epochs = 30
    batch_size = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = JEPADataset(
        dataset_path="src/dataset",  # TODO : adapt to your path
        labels_filename="labels.csv",
    )

    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset, val_dataset = random_split(
        dataset, [0.8, 0.1, 0.1], generator=generator
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        accuracy, train_loss, train_samples = train_linear_probing(
            test_loader, device=device
        )

        print(
            f"Epoch {epoch + 1}/{epochs} |",
            "train_acc:",
            accuracy,
            "train_loss:",
            (train_loss / train_samples),
        )
