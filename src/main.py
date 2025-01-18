import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from src.dataset.dataset import AffectNetDataset
from src.model.model import IJEPAModel
from src.utils.utils import image_to_patches


def train_ijepa():
    dataset = AffectNetDataset(
        dataset_path="dataset/affectnet",
        labels_filename="labels.csv",
        img_size=96,
        patch_size=8,
        nb_mask=4
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = IJEPAModel(embed_dim=768, nb_masks=4, patch_size=8).cuda()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    loss_fn = nn.MSELoss()

    for epoch in range(3):
        model.train()
        epoch_loss = 0.0
        k = 0

        for images, context_indices, masks in dataloader:
            images = images.cuda()
            context_indices = context_indices.cuda()
            masks = masks.cuda()

            ctx_emb, tgt_emb, recon = model(images, context_indices, masks)

            original_patches = image_to_patches(images, patch_size=model.patch_size)
            reconstructed_patches = image_to_patches(recon, patch_size=model.patch_size)

            pixel_recon_loss = 0.0
            pixel_count = 0  # to average the pixel loss if desired

            B = images.shape[0]  # batch size

            for m_i in range(masks.shape[1]):
                block_idxs = masks[:, m_i, :]  # (B, max_mask_len)
                for b in range(B):
                    valid_mask_idxs = block_idxs[b][block_idxs[b] != -1]  # the patch indices for this mask
                    for idx_ in valid_mask_idxs:
                        idx_int = idx_.item()
                        real_patch = original_patches[b, idx_int]  # => (3, 8, 8) if patch_size=8
                        pred_patch = reconstructed_patches[b, idx_int]  # => (3, 8, 8)

                        pixel_recon_loss += loss_fn(pred_patch, real_patch)
                        pixel_count += 1

            if pixel_count > 0:
                pixel_recon_loss = pixel_recon_loss / pixel_count

            embedding_loss = loss_fn(ctx_emb, tgt_emb)

            alpha = 5.0  # you can tune this
            total_loss = embedding_loss + alpha * pixel_recon_loss

            # 6) Backprop
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            if k % 100 == 0:
                print(total_loss.item())
            k += 1
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader):.4f}")

    model.eval()
    images, context_indices, masks = next(iter(dataloader))
    images = images.cuda()
    context_indices = context_indices.cuda()
    masks = masks.cuda()

    with torch.no_grad():
        _, _, reconstructed = model(images, context_indices, masks)

    orig_img = images[0].cpu().numpy().transpose(1, 2, 0)  # => (96,96,3)
    recon_img = reconstructed[0].cpu().numpy().transpose(1, 2, 0)  # => (96,96,3)

    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.imshow(orig_img)
    ax1.set_title("Original Image")
    ax2.imshow(recon_img)
    ax2.set_title("Reconstructed (Masked Patches filled)")
    plt.show()

    return model


if __name__ == "__main__":
    train_ijepa()
