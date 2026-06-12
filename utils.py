import cv2
import numpy as np


def get_edge_map(img_np):
    """Generate a Sobel edge map from an RGB image (numpy array)."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge = np.sqrt(sobelx ** 2 + sobely ** 2)
    if edge.max() == 0:
        return np.zeros_like(edge, dtype=np.uint8)
    return np.uint8(np.clip(edge / edge.max() * 255, 0, 255))


def get_seg_map(img_np):
    """Generate an HSV-based segmentation mask from an RGB image (numpy array)."""
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    return cv2.inRange(hsv, (0, 10, 0), (180, 255, 255))
