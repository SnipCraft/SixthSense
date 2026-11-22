
import sys
import cv2
import torch
import numpy as np
import time
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "Depth-Anything-V2"
CHECKPOINT = MODEL_DIR / "checkpoints" / "depth_anything_v2_vits.pth"

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

print("Depth model ready!")

# Initialize webcam
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not camera.isOpened():
    raise RuntimeError("Could not access webcam")

frame_count = 0
depth_colored = None

print("Live depth estimation started. Press Q to quit.")

try:
    while True:

        success, frame = camera.read()

        if not success:
            print("Failed to capture webcam frame.")
            break

        frame_count += 1

        # Run depth inference every 5 frames
        if frame_count % 5 == 0:

            # Resize input to reduce processing workload
            small_frame = cv2.resize(
                frame,
                (320, 240),
                interpolation=cv2.INTER_AREA
            )

            image_rgb = cv2.cvtColor(
                small_frame,
                cv2.COLOR_BGR2RGB
            )

            start_time = time.perf_counter()

            with torch.inference_mode():
                depth = model.infer_image(image_rgb)

            inference_time = time.perf_counter() - start_time

            print(f"Depth inference: {inference_time:.2f} seconds")

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

        # Display camera and latest depth map
        if depth_colored is not None:

            depth_display = cv2.resize(
                depth_colored,
                (frame.shape[1], frame.shape[0])
            )

            combined = np.hstack((frame, depth_display))

        else:
            combined = frame

        cv2.imshow(
            "SixthSense - Live Depth Estimation",
            combined
        )

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()

print("Live depth test completed!")
