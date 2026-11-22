
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "Depth-Anything-V2"
CHECKPOINT = MODEL_DIR / "checkpoints" / "depth_anything_v2_vits.pth"
IMAGE_PATH = BASE_DIR / "test_image.jpg"

# Add official model source to Python path
sys.path.insert(0, str(MODEL_DIR))

from depth_anything_v2.dpt import DepthAnythingV2

# Use CPU to avoid requiring an NVIDIA GPU
torch.set_num_threads(2)
device = "cpu"

# Small model configuration
model_config = {
    "encoder": "vits",
    "features": 64,
    "out_channels": [48, 96, 192, 384]
}

print("Loading Depth Anything V2 Small...")

model = DepthAnythingV2(**model_config)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(checkpoint)
model = model.to(device)
model.eval()

print("Model loaded successfully!")

# Load input image
image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("Generating depth map...")

# Estimate relative depth
with torch.inference_mode():
    depth = model.infer_image(image_rgb)

# Normalize for visualization
depth_normalized = cv2.normalize(
    depth,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)

# Save depth map
output_path = BASE_DIR / "depth_output.png"
cv2.imwrite(str(output_path), depth_normalized)

# Display original and depth map
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(depth_normalized, cmap="inferno")
plt.title("Estimated Relative Depth")
plt.axis("off")

plt.tight_layout()
plt.show()

print(f"Depth map saved at: {output_path}")
print("Depth estimation completed!")
