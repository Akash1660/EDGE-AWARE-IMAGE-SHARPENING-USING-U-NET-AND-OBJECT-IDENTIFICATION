"""
Inference/testing script for Edge-Aware Image Sharpening using U-Net.

Usage:
    python test.py --model_path checkpoints/unet_edge_seg.pt \
        --input_dir test_images --output_dir results/test_output
"""

import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model import UNetSmall
from utils import get_edge_map, get_seg_map


@torch.no_grad()
def test(model_path, input_dir, output_dir, show_images=False):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNetSmall().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    if show_images:
        import matplotlib.pyplot as plt

    for i, fname in enumerate(files):
        path = os.path.join(input_dir, fname)
        blur_np = np.array(Image.open(path).convert("RGB"))

        edge = get_edge_map(blur_np)
        seg = get_seg_map(blur_np)

        edge_t = torch.from_numpy(cv2.resize(edge, (256, 256))).unsqueeze(0).float() / 255.0
        seg_t = torch.from_numpy(cv2.resize(seg, (256, 256))).unsqueeze(0).float() / 255.0
        blur_t = transform(Image.fromarray(blur_np))

        inp = torch.cat([blur_t, edge_t, seg_t], dim=0).unsqueeze(0).to(device)
        pred = model(inp)[0]
        pred_np = ((pred.permute(1, 2, 0) * 0.5) + 0.5).clamp(0, 1).cpu().numpy()

        merged = np.concatenate([
            cv2.resize(blur_np, (256, 256)),
            (pred_np * 255).astype(np.uint8),
        ], axis=1)

        save_path = os.path.join(output_dir, f"deblur_{i}_{fname}")
        cv2.imwrite(save_path, cv2.cvtColor(merged, cv2.COLOR_RGB2BGR))
        print(f"[{i + 1}/{len(files)}] Saved: {save_path}")

        if show_images and i < 5:
            plt.figure(figsize=(10, 4))
            plt.imshow(merged)
            plt.axis('off')
            plt.title(f"Blurred | Deblurred - {fname}")
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="results/test_output")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    test(
        model_path=args.model_path,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        show_images=args.show,
    )
