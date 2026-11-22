
import time
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "depth_anything_v2_vits.onnx"

print("Checking ONNX model...")

# Validate ONNX graph
onnx.checker.check_model(str(MODEL_PATH))

print("ONNX model structure is valid!")

# Configure CPU inference
options = ort.SessionOptions()
options.graph_optimization_level = (
    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
)
options.intra_op_num_threads = 2
options.inter_op_num_threads = 1

print("Loading ONNX Runtime session...")

session = ort.InferenceSession(
    str(MODEL_PATH),
    sess_options=options,
    providers=["CPUExecutionProvider"]
)

print("ONNX Runtime session ready!")

# Inspect model inputs and outputs
input_info = session.get_inputs()[0]
output_info = session.get_outputs()[0]

print("Input name:", input_info.name)
print("Input shape:", input_info.shape)
print("Output name:", output_info.name)
print("Output shape:", output_info.shape)

# Test inference using a dummy normalized tensor
dummy_input = np.random.randn(
    1, 3, 518, 518
).astype(np.float32)

print("Running test inference...")

start_time = time.perf_counter()

output = session.run(
    [output_info.name],
    {input_info.name: dummy_input}
)[0]

inference_time = time.perf_counter() - start_time

print("Inference successful!")
print("Output shape:", output.shape)
print("Output contains finite values:", np.isfinite(output).all())
print(f"ONNX inference time: {inference_time:.2f} seconds")
