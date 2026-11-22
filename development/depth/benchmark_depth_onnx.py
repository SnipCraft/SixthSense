
import sys
import time
import numpy as np
import torch
import onnxruntime as ort
from pathlib import Path

# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "Depth-Anything-V2"

CHECKPOINT = (
    MODEL_DIR / "checkpoints" / "depth_anything_v2_vits.pth"
)

ONNX_PATH = BASE_DIR / "depth_anything_v2_vits.onnx"

sys.path.insert(0, str(MODEL_DIR))

from depth_anything_v2.dpt import DepthAnythingV2


# ============================================================
# 2. CPU CONFIGURATION
# ============================================================

# Use identical CPU thread settings
torch.set_num_threads(2)


# ============================================================
# 3. LOAD PYTORCH MODEL
# ============================================================

model_config = {
    "encoder": "vits",
    "features": 64,
    "out_channels": [48, 96, 192, 384]
}

print("=" * 55)
print("LOADING PYTORCH MODEL")
print("=" * 55)

torch_model = DepthAnythingV2(**model_config)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=True
)

torch_model.load_state_dict(checkpoint)
torch_model.eval()

print("PyTorch model loaded successfully.")

print(
    "PyTorch model device:",
    next(torch_model.parameters()).device
)


# ============================================================
# 4. LOAD ONNX RUNTIME MODEL
# ============================================================

print("\n" + "=" * 55)
print("LOADING ONNX RUNTIME MODEL")
print("=" * 55)

options = ort.SessionOptions()

options.graph_optimization_level = (
    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
)

options.intra_op_num_threads = 2
options.inter_op_num_threads = 1

onnx_session = ort.InferenceSession(
    str(ONNX_PATH),
    sess_options=options,
    providers=["CPUExecutionProvider"]
)

input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

print("ONNX model loaded successfully.")


# ============================================================
# 5. ONNX MODEL INFORMATION
# ============================================================

print("\n" + "=" * 55)
print("ONNX MODEL INFORMATION")
print("=" * 55)

print(
    "Input name:",
    input_name
)

print(
    "Input shape:",
    onnx_session.get_inputs()[0].shape
)

print(
    "Input type:",
    onnx_session.get_inputs()[0].type
)

print(
    "Output name:",
    output_name
)

print(
    "Output shape:",
    onnx_session.get_outputs()[0].shape
)


# ============================================================
# 6. CREATE IDENTICAL INPUT FOR BOTH MODELS
# ============================================================

print("\n" + "=" * 55)
print("PREPARING TEST INPUT")
print("=" * 55)

np.random.seed(42)

input_array = np.random.randn(
    1, 3, 518, 518
).astype(np.float32)

torch_input = torch.from_numpy(input_array)

print("Input shape:", input_array.shape)
print("Input dtype:", input_array.dtype)


# ============================================================
# 7. WARM-UP BOTH MODELS
# ============================================================

print("\n" + "=" * 55)
print("WARMING UP MODELS")
print("=" * 55)

with torch.inference_mode():

    for _ in range(2):
        torch_output = torch_model(torch_input)


for _ in range(2):

    onnx_output = onnx_session.run(
        [output_name],
        {input_name: input_array}
    )[0]

print("Warm-up completed.")


# ============================================================
# 8. BENCHMARK PYTORCH
# ============================================================

print("\n" + "=" * 55)
print("BENCHMARKING PYTORCH")
print("=" * 55)

torch_times = []

with torch.inference_mode():

    for i in range(5):

        start = time.perf_counter()

        torch_output = torch_model(torch_input)

        elapsed = time.perf_counter() - start

        torch_times.append(elapsed)

        print(
            f"PyTorch run {i + 1}: "
            f"{elapsed:.3f} seconds"
        )


# ============================================================
# 9. BENCHMARK ONNX RUNTIME
# ============================================================

print("\n" + "=" * 55)
print("BENCHMARKING ONNX RUNTIME")
print("=" * 55)

onnx_times = []

for i in range(5):

    start = time.perf_counter()

    onnx_output = onnx_session.run(
        [output_name],
        {input_name: input_array}
    )[0]

    elapsed = time.perf_counter() - start

    onnx_times.append(elapsed)

    print(
        f"ONNX run {i + 1}: "
        f"{elapsed:.3f} seconds"
    )


# ============================================================
# 10. COMPARE MEDIAN INFERENCE TIMES
# ============================================================

torch_median = float(np.median(torch_times))
onnx_median = float(np.median(onnx_times))

print("\n" + "=" * 55)
print("BENCHMARK RESULTS")
print("=" * 55)

print(
    f"PyTorch median: {torch_median:.3f} seconds"
)

print(
    f"ONNX median:    {onnx_median:.3f} seconds"
)

if onnx_median < torch_median:

    speedup = torch_median / onnx_median

    print(
        f"ONNX speedup: {speedup:.2f}x"
    )

else:

    slowdown = onnx_median / torch_median

    print(
        f"ONNX slowdown: {slowdown:.2f}x"
    )


# ============================================================
# 11. COMPARE MODEL OUTPUTS
# ============================================================

torch_result = (
    torch_output.detach().cpu().numpy()
)

difference = np.abs(
    torch_result - onnx_output
)

print("\n" + "=" * 55)
print("OUTPUT COMPARISON")
print("=" * 55)

print(
    "PyTorch output shape:",
    torch_result.shape
)

print(
    "ONNX output shape:",
    onnx_output.shape
)

print(
    f"Mean absolute difference: "
    f"{difference.mean():.6f}"
)

print(
    f"Maximum absolute difference: "
    f"{difference.max():.6f}"
)


# ============================================================
# 12. BENCHMARK COMPLETION
# ============================================================

print("\n" + "=" * 55)
print("BENCHMARK COMPLETED SUCCESSFULLY!")
print("=" * 55)
