
import cv2
import numpy as np
import torch
import onnxruntime as ort
import time

from pathlib import Path
from ultralytics import YOLO


# --------------------------------------------------
# 1. PROJECT PATHS AND SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

YOLO_PATH = PROJECT_ROOT / "yolo11n.pt"
DEPTH_ONNX_PATH = BASE_DIR / "depth" / "depth_anything_v2_vits_392.onnx"

INPUT_SIZE = 392
PROCESS_EVERY_N_FRAMES = 5


# --------------------------------------------------
# 2. LOAD AI MODELS
# --------------------------------------------------

print("\n========================================")
print("       SIXTHSENSE INTEGRATION TEST")
print("========================================")

if not YOLO_PATH.exists():
    raise FileNotFoundError(f"YOLO model not found: {YOLO_PATH}")

if not DEPTH_ONNX_PATH.exists():
    raise FileNotFoundError(f"Depth model not found: {DEPTH_ONNX_PATH}")

torch.set_num_threads(2)

print("\nLoading YOLO model...")
yolo_model = YOLO(str(YOLO_PATH))
print("YOLO loaded successfully!")

print("\nLoading Depth Anything ONNX model...")

options = ort.SessionOptions()
options.intra_op_num_threads = 2
options.inter_op_num_threads = 1
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

depth_session = ort.InferenceSession(
    str(DEPTH_ONNX_PATH),
    sess_options=options,
    providers=["CPUExecutionProvider"]
)

print("Depth model loaded successfully!")


# --------------------------------------------------
# 3. DEPTH PREPROCESSING
# --------------------------------------------------

def preprocess_depth(frame):

    height, width = frame.shape[:2]

    scale = min(INPUT_SIZE / width, INPUT_SIZE / height)

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_CUBIC
    )

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    image = rgb.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    image = (image - mean) / std

    pad_left = (INPUT_SIZE - new_width) // 2
    pad_top = (INPUT_SIZE - new_height) // 2

    padded = np.zeros(
        (INPUT_SIZE, INPUT_SIZE, 3),
        dtype=np.float32
    )

    padded[
        pad_top:pad_top + new_height,
        pad_left:pad_left + new_width
    ] = image

    input_array = np.transpose(padded, (2, 0, 1))
    input_array = np.expand_dims(input_array, axis=0)

    padding_info = (
        pad_left,
        pad_top,
        new_width,
        new_height
    )

    return input_array, padding_info


# --------------------------------------------------
# 4. DEPTH ESTIMATION
# --------------------------------------------------

def estimate_depth(frame):

    input_array, padding_info = preprocess_depth(frame)

    input_name = depth_session.get_inputs()[0].name
    output_name = depth_session.get_outputs()[0].name

    output = depth_session.run(
        [output_name],
        {input_name: input_array}
    )[0]

    depth = np.squeeze(output)

    pad_left, pad_top, new_width, new_height = padding_info

    depth = depth[
        pad_top:pad_top + new_height,
        pad_left:pad_left + new_width
    ]

    height, width = frame.shape[:2]

    depth = cv2.resize(
        depth,
        (width, height),
        interpolation=cv2.INTER_LINEAR
    )

    return depth


# --------------------------------------------------
# 5. START WEBCAM
# --------------------------------------------------

print("\nConnecting webcam...")

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    raise RuntimeError("Could not access webcam.")

print("Webcam connected!")
print("\nPress Q to exit.")


# --------------------------------------------------
# 6. RUN BOTH MODELS ON THE SAME FRAME
# --------------------------------------------------

frame_count = 0
combined_view = None

try:

    while True:

        success, frame = camera.read()

        if not success:
            print("Could not capture webcam frame.")
            break

        frame_count += 1

        # Run both models periodically to reduce CPU load.
        if (frame_count - 1) % PROCESS_EVERY_N_FRAMES == 0:

            start_time = time.time()

            # Object detection
            yolo_results = yolo_model.predict(
                frame,
                imgsz=416,
                conf=0.25,
                device="cpu",
                verbose=False
            )

            annotated_frame = yolo_results[0].plot()

            # Depth estimation on the SAME original frame
            depth_map = estimate_depth(frame)

            # Convert depth map into a displayable image
            depth_normalized = cv2.normalize(
                depth_map,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            )

            depth_normalized = depth_normalized.astype(np.uint8)

            depth_colored = cv2.applyColorMap(
                depth_normalized,
                cv2.COLORMAP_INFERNO
            )

            total_time = time.time() - start_time

            # Display both outputs side by side
            combined_view = np.hstack(
                (annotated_frame, depth_colored)
            )

            cv2.putText(
                combined_view,
                f"Combined inference: {total_time:.2f} seconds",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            print(
                f"Combined inference time: {total_time:.2f}s"
            )

        if combined_view is not None:

            cv2.imshow(
                "SixthSense - YOLO + Depth Integration",
                combined_view
            )

        else:

            cv2.imshow("SixthSense - Starting...", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    camera.release()
    cv2.destroyAllWindows()

    print("\nSixthSense integration test completed!")
