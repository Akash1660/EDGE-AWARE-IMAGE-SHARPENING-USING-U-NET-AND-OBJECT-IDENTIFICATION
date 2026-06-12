"""
Gradio web UI for Edge-Aware Image Sharpening using U-Net.

Usage:
    python app.py
(Set MODEL_PATH below or via environment variable before running.)
"""

import os

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model import UNetSmall
from utils import get_edge_map, get_seg_map

MODEL_PATH = os.environ.get("MODEL_PATH", "checkpoints/unet_edge_seg.pt")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNetSmall().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


def normalize_tensor(t):
    return (t - 0.5) / 0.5


def tensor_to_image(tensor):
    arr = ((tensor.permute(1, 2, 0).cpu().numpy() * 0.5) + 0.5).clip(0, 1)
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)


def sharpen_image(input_img):
    img = np.array(input_img.convert("RGB"))
    resized = cv2.resize(img, (256, 256))

    edge = get_edge_map(resized)
    seg = get_seg_map(resized)

    blur_t = normalize_tensor(transforms.ToTensor()(Image.fromarray(resized)))
    edge_t = torch.from_numpy(edge).unsqueeze(0).float() / 255.0
    seg_t = torch.from_numpy(seg).unsqueeze(0).float() / 255.0

    inp = torch.cat([blur_t, edge_t, seg_t], dim=0).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(inp)[0]

    return tensor_to_image(out)


title = "Image Sharpening using U-Net"
description = "Upload a blurred image to view its sharpened version."

demo = gr.Interface(
    fn=sharpen_image,
    inputs=gr.Image(type="pil", label="Upload Blurred Image"),
    outputs=gr.Image(type="pil", label="Sharpened Output"),
    title=title,
    description=description,
)

if __name__ == "__main__":
    demo.launch()
