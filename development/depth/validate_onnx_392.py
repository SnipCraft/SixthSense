
import sys
import time
import numpy as np
import torch
import onnxruntime as ort
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

ONNX_PATH = (
    BASE_DIR / "depth_anything_v2_vits_392.onnx"
)

INPUT_SIZE = 392

sys.path.insert(0, str(MODEL_DIR))

from depth_anything_v2.dpt import DepthAnythingV2


# =====================================================
# 2. CONFIGURATION
# =====================================================

torch.set_num_threads(2)

model_config = {
    "encoder": "vits",
    "features": 64,
    "out_channels": [48, 96, 192, 384]
}


# =====================================================
# 3. LOAD PYTORCH MODEL
# =====================================================

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

print("PyTorch model loaded.")


# =====================================================
# 4. LOAD ONNX MODEL
# =====================================================

print("\nLOADING ONNX MODEL")

options = ort.SessionOptions()

options.graph_optimization_level = (
    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
)

options.intra_op_num_threads = 2
options.inter_op_num_threads = 1

session = ort.InferenceSession(
    str(ONNX_PATH),
    sess_options=options,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("ONNX model loaded.")

print("Input shape:", session.get_inputs()[0].shape)
print("Output shape:", session.get_outputs()[0].shape)


# =====================================================
# 5. CREATE IDENTICAL INPUT
# =====================================================

print("\nPREPARING INPUT")

np.random.seed(42)

input_array = np.random.randn(
    1, 3, INPUT_SIZE, INPUT_SIZE
).astype(np.float32)

torch_input = torch.from_numpy(input_array)

print("Input shape:", input_array.shape)


# =====================================================
# 6. WARM-UP
# =====================================================

print("\nWARMING UP")

with torch.inference_mode():

    for _ in range(2):
        torch_output = torch_model(torch_input)

for _ in range(2):

    onnx_output = session.run(
        [output_name],
        {input_name: input_array}
    )[0]

print("Warm-up completed.")


# =====================================================
# 7. PYTORCH BENCHMARK
# =====================================================

print("\nBENCHMARKING PYTORCH")

torch_times = []

with torch.inference_mode():

    for i in range(5):

        start = time.perf_counter()

        torch_output = torch_model(torch_input)

        elapsed = time.perf_counter() - start

        torch_times.append(elapsed)

        print(f"PyTorch run {i + 1}: {elapsed:.3f}s")


# =====================================================
# 8. ONNX BENCHMARK
# =====================================================

print("\nBENCHMARKING ONNX")

onnx_times = []

for i in range(5):

    start = time.perf_counter()

    onnx_output = session.run(
        [output_name],
        {input_name: input_array}
    )[0]

    elapsed = time.perf_counter() - start

    onnx_times.append(elapsed)

    print(f"ONNX run {i + 1}: {elapsed:.3f}s")


# =====================================================
# 9. PERFORMANCE COMPARISON
# =====================================================

torch_median = float(np.median(torch_times))
onnx_median = float(np.median(onnx_times))

print("\n" + "=" * 55)
print("PERFORMANCE RESULTS")
print("=" * 55)

print(f"PyTorch median: {torch_median:.3f}s")
print(f"ONNX median:    {onnx_median:.3f}s")

print(f"PyTorch FPS: {1 / torch_median:.2f}")
print(f"ONNX FPS:    {1 / onnx_median:.2f}")

if onnx_median < torch_median:

    print(
        f"ONNX speedup: "
        f"{torch_median / onnx_median:.2f}x"
    )


# =====================================================
# 10. NUMERICAL ACCURACY
# =====================================================

torch_result = torch_output.detach().cpu().numpy()

difference = np.abs(
    torch_result - onnx_output
)

print("\n" + "=" * 55)
print("OUTPUT COMPARISON")
print("=" * 55)

print("PyTorch output:", torch_result.shape)
print("ONNX output:", onnx_output.shape)

print(
    f"Mean absolute difference: "
    f"{difference.mean():.8f}"
)

print(
    f"Maximum absolute difference: "
    f"{difference.max():.8f}"
)

print("\nVALIDATION COMPLETED")
