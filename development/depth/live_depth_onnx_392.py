
import cv2
import numpy as np
import onnxruntime as ort
import time

from pathlib import Path

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

ONNX_PATH = BASE_DIR / "depth_anything_v2_vits_392.onnx"

INPUT_SIZE = 392

# --------------------------------------------------
# ONNX CPU CONFIGURATION
# --------------------------------------------------

options = ort.SessionOptions()

options.intra_op_num_threads = 2
options.inter_op_num_threads = 1

options.graph_optimization_level = (
    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
)

print("Loading Depth Anything V2 ONNX model...")

session = ort.InferenceSession(
    str(ONNX_PATH),
    sess_options=options,
    providers=["CPUExecutionProvider"]
)

print("ONNX model ready!")

# --------------------------------------------------
# WEBCAM INITIALIZATION
# --------------------------------------------------

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not camera.isOpened():
    raise RuntimeError("Could not access webcam")

# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

def preprocess_frame(frame):

    height, width = frame.shape[:2]

    # BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    rgb = rgb.astype(np.float32) / 255.0

    # Preserve aspect ratio
    scale = min(
        INPUT_SIZE / width,
        INPUT_SIZE / height
    )

    new_w = int(round(width * scale))
    new_h = int(round(height * scale))

    resized = cv2.resize(
        rgb,
        (new_w, new_h),
        interpolation=cv2.INTER_CUBIC
    )

    # ImageNet normalization
    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    )

    normalized = (resized - mean) / std

    # Center padding
    canvas = np.zeros(
        (INPUT_SIZE, INPUT_SIZE, 3),
        dtype=np.float32
    )

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

    padding = {
        "top": top,
        "left": left,
        "new_h": new_h,
        "new_w": new_w
    }

    return input_array, padding


# --------------------------------------------------
# DEPTH INFERENCE
# --------------------------------------------------

def estimate_depth(frame):

    input_array, padding = preprocess_frame(frame)

    start_time = time.perf_counter()

    depth = session.run(
        ["depth"],
        {"image": input_array}
    )[0]

    inference_time = time.perf_counter() - start_time

    depth = np.squeeze(depth)

    # Remove padded regions
    top = padding["top"]
    left = padding["left"]

    new_h = padding["new_h"]
    new_w = padding["new_w"]

    depth = depth[
        top:top + new_h,
        left:left + new_w
    ]

    # Resize depth map to camera frame dimensions
    height, width = frame.shape[:2]

    depth = cv2.resize(
        depth,
        (width, height),
        interpolation=cv2.INTER_LINEAR
    )

    return depth, inference_time


# --------------------------------------------------
# WEBCAM LOOP
# --------------------------------------------------

frame_count = 0

depth_colored = None

last_inference_time = 0.0
last_inference_fps = 0.0

print("\nLive ONNX depth estimation started.")
print("Press Q to quit.")

try:

    while True:

        success, frame = camera.read()

        if not success:
            print("Failed to capture webcam frame.")
            break

        frame_count += 1

        # Run inference every 5 frames
        if frame_count % 5 == 0:

            depth, inference_time = estimate_depth(frame)

            last_inference_time = inference_time

            if inference_time > 0:
                last_inference_fps = 1.0 / inference_time

            # Normalize depth map
            depth_normalized = cv2.normalize(
                depth,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            ).astype(np.uint8)

            # Apply color map
            depth_colored = cv2.applyColorMap(
                depth_normalized,
                cv2.COLORMAP_INFERNO
            )

            print(
                f"ONNX inference: {inference_time:.3f}s | "
                f"Inference FPS: {last_inference_fps:.2f}"
            )

        # Display camera and latest depth map
        if depth_colored is not None:

            combined = np.hstack(
                (frame, depth_colored)
            )

        else:
            combined = frame

        # Display performance information
        cv2.putText(
            combined,
            f"Inference: {last_inference_time:.3f}s",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            combined,
            f"Inference FPS: {last_inference_fps:.2f}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "SixthSense - Live ONNX Depth Estimation",
            combined
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    camera.release()
    cv2.destroyAllWindows()

print("\nLive ONNX depth test completed!")
