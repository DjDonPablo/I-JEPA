import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from model.i_jepa import ViTEncoder
from src.dataset.dataset import JEPADataset
from src.mask.multiblock import MaskCollator as MBMaskCollator
from torch.optim import lr_scheduler

from src.utils.utils import LinearWeightDecay


def train(
    model: ViTEncoder,
    loader: DataLoader,
    optimizer: optim.AdamW,
    criterion: nn.MSELoss,
    device: str,
):
    model.train()
    train_loss = 0.0
    train_samples = 0

    for images, masks_enc, masks_pred in loader:
        images = images.to(device)
        masks_enc = [u.to(device) for u in masks_enc]
        masks_pred = [u.to(device) for u in masks_pred]

        # TODO : pass through each model
        outputs = model(images, masks_enc)
        loss = criterion(outputs, []) # TODO : loss between embeddings from predictor and target, for loop on the 4 masks

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_samples += len(images)

    return train_loss, train_samples


def evaluate(
    model: ViTEncoder,
    loader: DataLoader,
    optimizer: optim.AdamW,
    criterion: nn.MSELoss,
    device: str,
):
    model.eval()
    val_loss = 0.0
    val_samples = 0

    for images, masks_enc, masks_pred in loader:
        images = images.to(device)
        masks_enc = [u.to(device) for u in masks_enc]
        masks_pred = [u.to(device) for u in masks_pred]

        # TODO : pass through each model
        outputs = model(images, masks_enc)
        loss = criterion(outputs, []) # TODO : loss between embeddings from predictor and target, for loop on the 4 masks

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        val_loss += loss.item()
        val_samples += len(images)

    return val_loss, val_samples


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # hyper parameters

    image_size = 96
    input_size = (image_size, image_size)
    patch_size = 12

    target_mask_scale = (0.15, 0.2)
    context_mask_scale = (0.85, 1.0)
    aspect_ratio = (0.75, 1.5)
    num_context_patch = 1
    num_target_patch = 4

    batch_size = 64
    epochs = 10
    learning_rate = 1e-4

    in_channels = 3
    embed_dim = 192  # vit_tiny
    num_heads = 6
    num_layers = 6
    num_classes = (
        0  # Let this param to 0, (it will keep the size of the generated embeddings)
    )

    #
    # dataset, dataloader
    #

    mask_collator = MBMaskCollator(
        input_size=input_size,
        patch_size=patch_size,
        pred_mask_scale=target_mask_scale,
        enc_mask_scale=context_mask_scale,
        aspect_ratio=aspect_ratio,
        nenc=num_context_patch,
        npred=num_target_patch,
        allow_overlap=False,
        min_keep=4,
    )

    dataset = JEPADataset(
        dataset_path="dataset/archive",  # TODO : adapt to your path
        labels_filename="labels.csv",
    )

    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset, val_dataset = random_split(dataset, [0.8, 0.1, 0.1], generator=generator)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=mask_collator, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=mask_collator, shuffle=True
    )
    val_loader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=mask_collator, shuffle=True
    )

    #
    # model
    #
    print("Loading models...")
    terminator = ViTEncoder()

    #
    # optimizer, scheduler, loss
    #
    criterion = nn.MSELoss(reduce="sum")
    optimizer = optim.AdamW(terminator.parameters(), lr=learning_rate, weight_decay=0.04)

    weight_decay_scheduler = LinearWeightDecay(adamw=optimizer, initial_wd=0.04, end_wd=0.4, num_steps=epochs)

    scheduler1 = lr_scheduler.MultiplicativeLR(
        optimizer, lr_lambda=lambda x: x * 1.16585
    )
    scheduler2 = lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-6)
    scheduler = lr_scheduler.SequentialLR(
        optimizer, schedulers=[scheduler1, scheduler2], milestones=[15]
    )

    #
    # loop
    #
    best_val_loss = 10e6
    for epoch in epochs:
        train_loss, train_samples = train(criterion=criterion, device=device, loader=test_loader, model=terminator, optimizer=optimizer)
        val_loss, val_samples = eval(criterion=criterion, device=device, loader=test_loader, model=terminator, optimizer=optimizer)

        print(
            f"Epoch [{epoch + 1}/{epoch}] - train loss: {train_loss / train_samples:4f.}, val loss: {val_loss / val_samples:4f.}, lr: {scheduler.get_last_lr()[0]}"
        )
        if val_loss / val_samples < best_val_loss:
            best_val_loss = val_loss / val_samples
            checkpoint = {
                "epoch": epoch,
                "model": terminator.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_sched": scheduler.get_last_lr()[0],
            }
            torch.save(checkpoint, "checkpoint.pth")

        scheduler.step()
        weight_decay_scheduler.step()
