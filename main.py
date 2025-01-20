import numpy as np
import os
import torch.nn as nn
import torch
import torch.optim as optim

from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split
from src.new_dataset import CIFAR10Dataset
from src.mask.multiblock import MaskCollator as MBMaskCollator
from src.utils.utils import LinearWeightDecay
from src.model.ijepa import IJEPA
from tqdm import tqdm
from linear_probing import LinearProbe


def train(
    model: IJEPA,
    loader: DataLoader,
    optimizer: optim.AdamW,
    criterion: nn.MSELoss,
    device: str,
    momentum_scheduler,
):
    model.train()
    train_loss = 0.0
    train_samples = 0

    for images, masks_enc, masks_pred in tqdm(loader):
        images = images.to(device)
        masks_enc = masks_enc[0].to(device)
        masks_pred = [mask.to(device) for mask in masks_pred]

        mask_preds, target_preds = model(images, masks_enc, masks_pred)

        loss = criterion(mask_preds, target_preds)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            m = next(momentum_scheduler)
            for param_q, param_k in zip(
                model.context_encoder.parameters(), model.target_encoder.parameters()
            ):
                param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

        train_loss += loss.item()
        train_samples += len(images)

    return train_loss, train_samples


def evaluate(
    model: IJEPA,
    loader: DataLoader,
    criterion: nn.MSELoss,
    device: str,
):
    model.eval()
    val_loss = 0.0
    val_samples = 0

    with torch.no_grad():
        for images, masks_enc, masks_pred in loader:
            images = images.to(device)
            masks_enc = masks_enc[0].to(device)
            masks_pred = [mask.to(device) for mask in masks_pred]

            mask_preds, target_preds = model(images, masks_enc, masks_pred)

            loss = criterion(mask_preds, target_preds)

            val_loss += loss.item()
            val_samples += images.size(0)

    return val_loss, val_samples


def lr_scheduler(optimizer, warmup_epochs, total_epochs, initial_lr, peak_lr, final_lr):
    def lr_lambda(epoch):
        if epoch == 0:
            return 1.0
        if epoch < warmup_epochs:
            # Augmentation linéaire du LR pendant la période de warm-up
            return initial_lr + (peak_lr - initial_lr) * (epoch / warmup_epochs)
        else:
            # Décroissance cosinus après le warm-up
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return final_lr + (peak_lr - final_lr) * 0.5 * (
                1 + np.cos(np.pi * progress)
            )

    return LambdaLR(optimizer, lr_lambda)


if __name__ == "__main__":
    #
    # hyper parameters
    #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_size = 32
    input_size = (image_size, image_size)
    patch_size = 4

    target_mask_scale = (0.15, 0.2)
    context_mask_scale = (0.85, 1.0)
    aspect_ratio = (0.75, 1.5)
    num_context_patch = 1
    num_target_patch = 4

    batch_size = 128
    epochs = 100
    learning_rate = 1e-4

    in_channels = 3
    embed_dim = 128
    num_heads = 6
    num_layers = 5

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

    # for i-jepa
    data_path = "/kaggle/working/"  # os.path.join("..", "cifar-10")
    train_dataset = CIFAR10Dataset(data_path, "unsupervised", "train")
    test_dataset = CIFAR10Dataset(data_path, "unsupervised", "test")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=mask_collator, shuffle=True
    )
    val_loader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=mask_collator, shuffle=True
    )

    # for linear model
    linear_train_dataset = CIFAR10Dataset(data_path, "supervised", "train")
    linear_test_dataset = CIFAR10Dataset(data_path, "supervised", "test")
    linear_train_loader = DataLoader(
        linear_train_dataset, batch_size=batch_size, shuffle=True
    )
    linear_val_loader = DataLoader(
        linear_test_dataset, batch_size=batch_size, shuffle=True
    )

    #
    # model
    #
    model = IJEPA(
        nb_mask=num_target_patch,
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
    ).to(device)

    print("Number of parameters :", sum(p.numel() for p in model.parameters()))

    #
    # optimizer, scheduler, loss
    #
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.04)

    criterion = nn.MSELoss(reduction="sum")
    linear_criterion = nn.CrossEntropyLoss(reduction="sum")

    weight_decay_scheduler = LinearWeightDecay(
        adamw=optimizer, initial_wd=0.04, end_wd=0.2, num_steps=epochs
    )

    scheduler = lr_scheduler(optimizer, 20, epochs, 1.0, 5.0, 0.01)

    #
    # loop
    #
    best_val_loss = 10e6

    # Exponential moving average
    ema = (0.996, 1.0)
    ipe = len(train_loader)
    momentum_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / (ipe * epochs)
        for i in range(int(ipe * epochs) + 1)
    )

    train_loss_jepa = []
    for epoch in range(1, epochs + 1):
        train_loss, train_samples = train(
            device=device,
            loader=train_loader,
            model=model,
            optimizer=optimizer,
            momentum_scheduler=momentum_scheduler,
            criterion=criterion,
        )
        val_loss, val_samples = evaluate(
            model=model, loader=val_loader, device=device, criterion=criterion
        )

        print(
            f"Epoch [{epoch}/{epochs}] - train loss: {train_loss / train_samples}, val loss: {val_loss / val_samples}, lr: {scheduler.get_last_lr()[0]}, wd: {weight_decay_scheduler.last_weight_decay}"
        )

        train_loss_jepa.append(train_loss / train_samples)
        print(train_loss_jepa)

        if val_loss / val_samples < best_val_loss:
            best_val_loss = val_loss / val_samples
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_sched": scheduler.state_dict(),
            }
            torch.save(checkpoint, "checkpoint.pth")

        scheduler.step()
        weight_decay_scheduler.step()

        # ------------------------------- EVAL WITH LINEAR MODEL -----------------------------

        if epoch % 10 == 0:
            linear_model = LinearProbe(test_dataset.nb_classes, embed_dim, model).to(
                device
            )
            linear_optimizer = optim.Adam(linear_model.parameters(), 3e-4)
            linear_epochs = 15

            for lepoch in range(1, linear_epochs + 1):
                linear_model.train()
                train_loss = 0
                train_samples = 0
                train_acc = 0
                for data, label in linear_train_loader:
                    data, label = data.to(device), label.to(device)

                    preds = linear_model(data)

                    loss = linear_criterion(preds, label)

                    pred = preds.argmax(dim=1)
                    train_acc += (pred == torch.argmax(label, dim=1)).sum().item()

                    linear_optimizer.zero_grad()
                    loss.backward()
                    linear_optimizer.step()

                    train_loss += loss.item()
                    train_samples += len(data)

                linear_model.eval()
                eval_loss = 0
                eval_samples = 0
                eval_accuracy = 0
                for data, label in linear_val_loader:
                    data, label = data.to(device), label.to(device)

                    preds = linear_model(data)
                    loss = linear_criterion(preds, label)

                    pred = preds.argmax(dim=1)
                    eval_accuracy += (pred == label.argmax(dim=1)).sum().item()

                    eval_loss += loss.item()
                    eval_samples += len(data)

                print(
                    f"[LINEAR] Epoch [{lepoch}/{linear_epochs}] | train_acc: {train_acc / train_samples}, train_loss: {train_loss / train_samples} | val_acc: {eval_accuracy / eval_samples}, val_loss: {eval_loss / eval_samples}"
                )

            model.evaluation_on = False
