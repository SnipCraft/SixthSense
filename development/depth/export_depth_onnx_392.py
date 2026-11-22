
import sys
import torch
from pathlib import Path

# =====================================================
# 1. PATH CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "Depth-Anything-V2"

CHECKPOINT = (
    MODEL_DIR / "checkpoints" /
    "depth_anything_v2_vits.pth"
)

OUTPUT_PATH = (
    BASE_DIR / "depth_anything_v2_vits_392.onnx"
)

sys.path.insert(0, str(MODEL_DIR))

from depth_anything_v2.dpt import DepthAnythingV2


# =====================================================
# 2. CONFIGURATION
# =====================================================

torch.set_num_threads(2)

INPUT_SIZE = 392

model_config = {
    "encoder": "vits",
    "features": 64,
    "out_channels": [48, 96, 192, 384]
}


# =====================================================
# 3. LOAD MODEL
# =====================================================

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


# =====================================================
# 4. CREATE FIXED-SIZE INPUT
# =====================================================

dummy_input = torch.randn(
    1,
    3,
    INPUT_SIZE,
    INPUT_SIZE,
    dtype=torch.float32
)

print("Input shape:", tuple(dummy_input.shape))


# =====================================================
# 5. EXPORT FIXED-SIZE ONNX
# =====================================================

print("\nStarting 392x392 ONNX export...")

with torch.inference_mode():

    torch.onnx.export(
        model,
        (dummy_input,),
        str(OUTPUT_PATH),

        input_names=["image"],
        output_names=["depth"],

        opset_version=17,
        dynamo=False,

        # Fixed dimensions: no dynamic_axes
    )


# =====================================================
# 6. EXPORT COMPLETION
# =====================================================

print("\n" + "=" * 55)
print("392x392 EXPORT COMPLETED")
print("=" * 55)

print("Saved model:", OUTPUT_PATH)

print(
    "File size:",
    f"{OUTPUT_PATH.stat().st_size / (1024 * 1024):.2f} MB"
)
