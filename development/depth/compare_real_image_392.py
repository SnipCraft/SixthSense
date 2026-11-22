
import sys
import cv2
import torch
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt

from pathlib import Path

# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(BASE_DIR / "Depth-Anything-V2"))

from depth_anything_v2.dpt import DepthAnythingV2

CHECKPOINT = (
    BASE_DIR / "Depth-Anything-V2"
    / "checkpoints" / "depth_anything_v2_vits.pth"
)

ONNX_PATH = BASE_DIR / "depth_anything_v2_vits_392.onnx"
IMAGE_PATH = BASE_DIR / "test.jpg"

INPUT_SIZE = 392

torch.set_num_threads(2)

model_configs = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384]
    }
}

# --------------------------------------------------
# LOAD IMAGE
# --------------------------------------------------

print("\nLOADING IMAGE")

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

original_h, original_w = image.shape[:2]

print("Original image:", image.shape)

# --------------------------------------------------
# LOAD PYTORCH MODEL
# --------------------------------------------------

print("\nLOADING PYTORCH MODEL")

model = DepthAnythingV2(**model_configs["vits"])

model.load_state_dict(
    torch.load(CHECKPOINT, map_location="cpu")
)

model.eval()

print("PyTorch model loaded.")

# --------------------------------------------------
# ASPECT-RATIO-PRESERVING PREPROCESSING
# --------------------------------------------------

print("\nPREPROCESSING IMAGE")

# Convert BGR to RGB and normalize to [0, 1].
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
rgb = rgb.astype(np.float32) / 255.0

# Resize while preserving aspect ratio.
scale = min(
    INPUT_SIZE / original_w,
    INPUT_SIZE / original_h
)

new_w = int(round(original_w * scale))
new_h = int(round(original_h * scale))

resized = cv2.resize(
    rgb,
    (new_w, new_h),
    interpolation=cv2.INTER_CUBIC
)

# Normalize using Depth Anything V2 ImageNet statistics.
mean = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

std = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)

normalized = (resized - mean) / std

# Create a square canvas.
# Padding is applied in normalized space using zero values.
canvas = np.zeros(
    (INPUT_SIZE, INPUT_SIZE, 3),
    dtype=np.float32
)

# Center the resized image.
top = (INPUT_SIZE - new_h) // 2
left = (INPUT_SIZE - new_w) // 2

canvas[
    top:top + new_h,
    left:left + new_w
] = normalized

# HWC -> CHW -> NCHW
input_array = np.transpose(
    canvas,
    (2, 0, 1)
)[None, ...]

input_array = input_array.astype(np.float32)

print("Resized image:", (new_h, new_w))
print("Padding: top/bottom =", top, INPUT_SIZE - new_h - top)
print("Padding: left/right =", left, INPUT_SIZE - new_w - left)
print("Final input:", input_array.shape)

input_tensor = torch.from_numpy(input_array)

# --------------------------------------------------
# PYTORCH INFERENCE
# --------------------------------------------------

print("\nRUNNING PYTORCH")

with torch.no_grad():
    pytorch_output = model(input_tensor)

pytorch_depth = pytorch_output.squeeze().cpu().numpy()

# --------------------------------------------------
# ONNX INFERENCE
# --------------------------------------------------

print("\nLOADING ONNX MODEL")

session = ort.InferenceSession(
    str(ONNX_PATH),
    providers=["CPUExecutionProvider"]
)

print("ONNX model loaded.")

print("\nRUNNING ONNX")

onnx_output = session.run(
    ["depth"],
    {"image": input_array}
)[0]

onnx_depth = onnx_output.squeeze()

# --------------------------------------------------
# NUMERICAL COMPARISON
# --------------------------------------------------

print("\nOUTPUT SHAPES")
print("PyTorch:", pytorch_depth.shape)
print("ONNX:", onnx_depth.shape)

difference = np.abs(pytorch_depth - onnx_depth)

print("\nNUMERICAL COMPARISON")
print("Mean absolute difference:", np.mean(difference))
print("Maximum absolute difference:", np.max(difference))

# --------------------------------------------------
# REMOVE PADDING AND RESIZE FOR VISUALIZATION
# --------------------------------------------------

pytorch_cropped = pytorch_depth[
    top:top + new_h,
    left:left + new_w
]

onnx_cropped = onnx_depth[
    top:top + new_h,
    left:left + new_w
]

pytorch_visual = cv2.resize(
    pytorch_cropped,
    (original_w, original_h),
    interpolation=cv2.INTER_LINEAR
)

onnx_visual = cv2.resize(
    onnx_cropped,
    (original_w, original_h),
    interpolation=cv2.INTER_LINEAR
)

# --------------------------------------------------
# VISUALIZATION
# --------------------------------------------------

plt.figure(figsize=(16, 6))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(pytorch_visual, cmap="inferno")
plt.title("PyTorch Depth - 392")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(onnx_visual, cmap="inferno")
plt.title("ONNX Depth - 392")
plt.axis("off")

plt.tight_layout()
plt.show()

print("\nREAL IMAGE COMPARISON COMPLETED")
