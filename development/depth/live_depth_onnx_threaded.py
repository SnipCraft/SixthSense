
import cv2
import numpy as np
import onnxruntime as ort
import threading
import time

from pathlib import Path

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

ONNX_PATH = BASE_DIR / "depth_anything_v2_vits_392.onnx"

INPUT_SIZE = 392
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# --------------------------------------------------
# ONNX CONFIGURATION
# --------------------------------------------------

options = ort.SessionOptions()
options.intra_op_num_threads = 2
options.inter_op_num_threads = 1
options.graph_optimization_level = (
    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
)

print("Loading ONNX model...")

session = ort.InferenceSession(
    str(ONNX_PATH),
    sess_options=options,
    providers=["CPUExecutionProvider"]
)

print("ONNX model loaded successfully!")

# --------------------------------------------------
# SHARED VARIABLES
# --------------------------------------------------

lock = threading.Lock()
stop_event = threading.Event()

latest_frame = None
latest_frame_id = 0

latest_depth = None
latest_inference_time = 0.0
latest_inference_fps = 0.0

inference_count = 0

# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------

def preprocess_frame(frame):

    height, width = frame.shape[:2]

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

    input_array = np.transpose(
        canvas,
        (2, 0, 1)
    )[None, ...]

    padding = (top, left, new_h, new_w)

    return input_array.astype(np.float32), padding


# --------------------------------------------------
# DEPTH INFERENCE WORKER
# --------------------------------------------------

def depth_worker():

    global latest_depth
    global latest_inference_time
    global latest_inference_fps
    global inference_count

    last_processed_id = -1

    while not stop_event.is_set():

        # Get the newest available frame
        with lock:

            if latest_frame is None:
                frame = None
                frame_id = -1
            else:
                frame = latest_frame.copy()
                frame_id = latest_frame_id

        # Wait if no new frame is available
        if frame is None or frame_id == last_processed_id:
            stop_event.wait(0.005)
            continue

        last_processed_id = frame_id

        try:

            start_time = time.perf_counter()

            input_array, padding = preprocess_frame(frame)

            output = session.run(
                ["depth"],
                {"image": input_array}
            )[0]

            inference_time = time.perf_counter() - start_time

            depth = np.squeeze(output)

            # Remove padding
            top, left, new_h, new_w = padding

            depth = depth[
                top:top + new_h,
                left:left + new_w
            ]

            # Resize depth map to camera dimensions
            height, width = frame.shape[:2]

            depth = cv2.resize(
                depth,
                (width, height),
                interpolation=cv2.INTER_LINEAR
            )

            inference_fps = (
                1.0 / inference_time
                if inference_time > 0 else 0.0
            )

            with lock:

                latest_depth = depth
                latest_inference_time = inference_time
                latest_inference_fps = inference_fps
                inference_count += 1

            print(
                f"Inference: {inference_time:.3f}s | "
                f"Inference FPS: {inference_fps:.2f}"
            )

        except Exception as error:

            print("Inference error:", error)
            stop_event.set()
            break


# --------------------------------------------------
# WEBCAM INITIALIZATION
# --------------------------------------------------

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

if not camera.isOpened():
    raise RuntimeError("Could not access webcam")

# --------------------------------------------------
# START INFERENCE THREAD
# --------------------------------------------------

worker = threading.Thread(
    target=depth_worker,
    daemon=True
)

worker.start()

print("\nThreaded ONNX depth estimation started.")
print("Press Q to quit.")

# --------------------------------------------------
# MAIN CAMERA AND DISPLAY LOOP
# --------------------------------------------------

camera_frame_count = 0
camera_start_time = time.perf_counter()
camera_fps = 0.0

try:

    while True:

        success, frame = camera.read()

        if not success:
            print("Camera frame capture failed.")
            break

        camera_frame_count += 1

        # Publish newest frame to inference worker
        with lock:

            latest_frame = frame.copy()
            latest_frame_id += 1

            depth = (
                latest_depth.copy()
                if latest_depth is not None
                else None
            )

            inference_time = latest_inference_time
            inference_fps = latest_inference_fps

        # Calculate camera FPS
        elapsed = time.perf_counter() - camera_start_time

        if elapsed >= 1.0:
            camera_fps = camera_frame_count / elapsed
            camera_frame_count = 0
            camera_start_time = time.perf_counter()

        # Display depth map
        if depth is not None:

            depth_normalized = cv2.normalize(
                depth,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            ).astype(np.uint8)

            depth_colored = cv2.applyColorMap(
                depth_normalized,
                cv2.COLORMAP_INFERNO
            )

            combined = np.hstack((frame, depth_colored))

        else:

            combined = frame.copy()

        # Performance information
        cv2.putText(
            combined,
            f"Camera FPS: {camera_fps:.1f}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            combined,
            f"Inference FPS: {inference_fps:.2f}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            combined,
            f"Inference time: {inference_time:.3f}s",
            (15, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "SixthSense - Threaded ONNX Depth",
            combined
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    stop_event.set()
    worker.join(timeout=2.0)

    camera.release()
    cv2.destroyAllWindows()

print("\nThreaded ONNX depth test completed!")
