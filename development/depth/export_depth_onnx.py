
import sys
import torch
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "Depth-Anything-V2"
CHECKPOINT = MODEL_DIR / "checkpoints" / "depth_anything_v2_vits.pth"
OUTPUT_PATH = BASE_DIR / "depth_anything_v2_vits.onnx"

sys.path.insert(0, str(MODEL_DIR))

from depth_anything_v2.dpt import DepthAnythingV2

# CPU configuration
torch.set_num_threads(2)

# Model configuration
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
model.eval()

print("Model loaded successfully!")

# Dummy input: batch size 1, RGB, 518 x 518
dummy_input = torch.randn(1, 3, 518, 518)

print("Starting ONNX export...")

torch.onnx.export(
    model,
    (dummy_input,),
    str(OUTPUT_PATH),
    input_names=["image"],
    output_names=["depth"],
    opset_version=17,
    dynamo=False,
    dynamic_axes={
        "image": {0: "batch_size"},
        "depth": {0: "batch_size"}
    }
)

print("ONNX export completed!")
print(f"Saved model: {OUTPUT_PATH}")
print(f"File size: {OUTPUT_PATH.stat().st_size / (1024 * 1024):.2f} MB")
