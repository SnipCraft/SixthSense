
import time
import numpy as np
import onnxruntime as ort
from pathlib import Path

# =====================================================
# PATH CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

ONNX_PATH = (
    BASE_DIR / "depth_anything_v2_vits_dynamic.onnx"
)

# Test resolutions
RESOLUTIONS = [
    (518, 518),
    (392, 392),
    (280, 280),
    (224, 224)
]

# =====================================================
# LOAD ONNX MODEL
# =====================================================

print("=" * 55)
print("LOADING DYNAMIC ONNX MODEL")
print("=" * 55)

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

input_info = session.get_inputs()[0]
output_info = session.get_outputs()[0]

input_name = input_info.name
output_name = output_info.name

print("Model loaded successfully!")
print("Input shape:", input_info.shape)
print("Output shape:", output_info.shape)


# =====================================================
# TEST DIFFERENT RESOLUTIONS
# =====================================================

print("\n" + "=" * 55)
print("DYNAMIC RESOLUTION VALIDATION")
print("=" * 55)

for height, width in RESOLUTIONS:

    print(f"\nTesting resolution: {height}x{width}")

    try:

        # Create random input
        input_array = np.random.randn(
            1, 3, height, width
        ).astype(np.float32)

        # Warm-up
        session.run(
            [output_name],
            {input_name: input_array}
        )

        # Benchmark
        times = []
        output = None

        for _ in range(3):

            start = time.perf_counter()

            output = session.run(
                [output_name],
                {input_name: input_array}
            )[0]

            elapsed = time.perf_counter() - start
            times.append(elapsed)

        median_time = float(np.median(times))

        print("Status: SUCCESS")
        print("Input shape:", input_array.shape)
        print("Output shape:", output.shape)

        print(
            f"Median inference time: "
            f"{median_time:.3f} seconds"
        )

        print(
            f"Theoretical inference FPS: "
            f"{1 / median_time:.2f}"
        )

        if output.shape[-2:] == (height, width):
            print("Spatial dimensions: MATCH")
        else:
            print("Spatial dimensions: DO NOT MATCH")

    except Exception as error:

        print("Status: FAILED")
        print("Error:", str(error))


# =====================================================
# COMPLETION
# =====================================================

print("\n" + "=" * 55)
print("DYNAMIC VALIDATION COMPLETED")
print("=" * 55)
