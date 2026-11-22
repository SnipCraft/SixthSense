
import sys
import torch
from pathlib import Path

# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "Depth-Anything-V2"

CHECKPOINT = (
    MODEL_DIR / "checkpoints" / "depth_anything_v2_vits.pth"
)

OUTPUT_PATH = (
    BASE_DIR / "depth_anything_v2_vits_dynamic.onnx"
)

sys.path.insert(0, str(MODEL_DIR))

from depth_anything_v2.dpt import DepthAnythingV2


# ============================================================
# 2. CPU CONFIGURATION
# ============================================================

torch.set_num_threads(2)


# ============================================================
# 3. MODEL CONFIGURATION
# ============================================================

model_config = {
    "encoder": "vits",
    "features": 64,
    "out_channels": [48, 96, 192, 384]
}


# ============================================================
# 4. LOAD PYTORCH MODEL
# ============================================================

print("=" * 55)
print("LOADING DEPTH ANYTHING V2 SMALL")
print("=" * 55)

model = DepthAnythingV2(**model_config)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(checkpoint)
model.eval()

print("Model loaded successfully!")


# ============================================================
# 5. CREATE DUMMY INPUT
# ============================================================

dummy_input = torch.randn(
    1, 3, 518, 518,
    dtype=torch.float32
)

print("Dummy input shape:", tuple(dummy_input.shape))


# ============================================================
# 6. EXPORT DYNAMIC ONNX MODEL
# ============================================================

print("\nStarting dynamic ONNX export...")

torch.onnx.export(
    model,
    (dummy_input,),
    str(OUTPUT_PATH),

    input_names=["image"],
    output_names=["depth"],

    opset_version=17,
    dynamo=False,

    dynamic_axes={
        "image": {
            0: "batch_size",
            2: "height",
            3: "width"
        },

        "depth": {
            0: "batch_size",
            1: "height",
            2: "width"
        }
    }
)


# ============================================================
# 7. EXPORT COMPLETION
# ============================================================

print("\n" + "=" * 55)
print("DYNAMIC ONNX EXPORT COMPLETED")
print("=" * 55)

print("Saved model:", OUTPUT_PATH)

print(
    "File size:",
    f"{OUTPUT_PATH.stat().st_size / (1024 * 1024):.2f} MB"
)
