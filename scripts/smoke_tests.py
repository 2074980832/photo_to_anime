# scripts/smoke_test.py
import torch, diffusers
from PIL import Image
import cv2
import numpy as np
print("python", __import__('sys').version)
print("torch:", torch.__version__, "cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    try:
        print("nvidia-smi check (torch):", torch.cuda.get_device_name(0))
    except Exception as e:
        print("torch CUDA device name error:", e)
print("diffusers:", diffusers.__version__)
im = Image.new("RGB", (64,64), color=(255,255,255))
print("PIL OK, size:", im.size)
arr = np.zeros((10,10), dtype=np.uint8)
edges = cv2.Canny(arr, 50, 150)
print("cv2 OK, edges shape:", edges.shape)