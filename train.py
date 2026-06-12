"""
Training script for Edge-Aware Image Sharpening using U-Net with Knowledge Distillation.

Usage:
    python train.py --data_dir /path/to/images --save_path checkpoints/unet_edge_seg.pt \
        --output_dir results --epochs 300
"""

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from pytorch_msssim import ssim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model import UNetSmall
from utils import get_edge_map, get_seg_map


class BlurDataset(Dataset):
    """Loads clean images, applies synthetic Gaussian blur, and builds the
    5-channel (RGB + edge + segmentation) input tensor."""

    def __init__(self, folder, use_clahe=False):
        self.files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        self.use_clahe = use_clahe
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        self.resize = transforms.Resize((256, 256))

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        sharp_np = np.array(img)

        if self.use_clahe:
            lab = cv2.cvtColor(sharp_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            cl = cv2.createCLAHE(3.0, (8, 8)).apply(l)
            sharp_np = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)

        blur_np = cv2.GaussianBlur(sharp_np, (21, 21), 2.5)
        edge_np = get_edge_map(blur_np)
        seg_np = get_seg_map(blur_np)

        edge_t = torch.from_numpy(
            np.array(self.resize(Image.fromarray(edge_np)))
        ).unsqueeze(0).float() / 255.0

        seg_t = torch.from_numpy(
            np.array(self.resize(Image.fromarray(seg_np)))
        ).unsqueeze(0).float() / 255.0

        blur_tensor = self.to_tensor(self.resize(Image.fromarray(blur_np)))
        sharp_tensor = self.to_tensor(self.resize(Image.fromarray(sharp_np)))

        input_tensor = torch.cat([blur_tensor, edge_t, seg_t], dim=0)
        return input_tensor, sharp_tensor

    def __len__(self):
        return len(self.files)


def loss_fn(pred, target):
    """Hybrid loss: 0.4 * MSE + 0.6 * (1 - SSIM)"""
    mse = nn.functional.mse_loss(pred, target)
    ssim_loss = 1 - ssim(pred, target, data_range=2)
    return 0.4 * mse + 0.6 * ssim_loss


def psnr(pred, target):
    mse = nn.functional.mse_loss(pred, target).item()
    return 20 * np.log10(2.0) - 10 * np.log10(mse) if mse > 0 else 100


def train(data_dir, save_path, output_dir, epochs=300, use_clahe=True):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = BlurDataset(data_dir, use_clahe)
    if len(dataset) == 0:
        print(f"Error: No images found in {data_dir}")
        return

    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    model = UNetSmall().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-4)

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        print(f"Epoch {ep:03d}/{epochs} | Loss: {total_loss / len(loader):.5f}")

        if ep % 10 == 0:
            ckpt_name = save_path.replace(".pt", f"_ep{ep:03d}.pt")
            torch.save(model.state_dict(), ckpt_name)
            print("Model saved at:", ckpt_name)

    # Final evaluation
    model.eval()
    ssim_total, psnr_total, total_loss = 0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        with torch.no_grad():
            pred = model(xb)
            total_loss += loss_fn(pred, yb).item()
            ssim_total += ssim(pred, yb, data_range=2).item()
            psnr_total += psnr(pred, yb)

    n = len(loader)
    print(f"\nAvg SSIM: {ssim_total / n:.4f} | "
          f"Avg PSNR: {psnr_total / n:.2f} dB | "
          f"Avg Loss: {total_loss / n:.5f}")

    torch.save(model.state_dict(), save_path)
    print("Final model saved at:", save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Folder of training images")
    parser.add_argument("--save_path", default="checkpoints/unet_edge_seg.pt")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--no_clahe", action="store_true")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        save_path=args.save_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        use_clahe=not args.no_clahe,
    )
