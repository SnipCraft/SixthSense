
# sixthsense_object_depth_priority_test.py
# SixthSense
# Object Detection + Depth Association + Obstacle Prioritization
# Region Status & Direction Guidance


import cv2
import numpy as np
import torch
import onnxruntime as ort
import time

from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

from obstacle_prioritization import prioritize_obstacles


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

YOLO_PATH = PROJECT_ROOT / "yolo11n.pt"

DEPTH_PATH = (
    BASE_DIR
    / "depth"
    / "depth_anything_v2_vits_392.onnx"
)

INPUT_SIZE = 392
PROCESS_EVERY_N_FRAMES = 5

CONFIDENCE_THRESHOLD = 0.50


# --------------------------------------------------
# 2. LOAD MODELS
# --------------------------------------------------

print("\n========================================")
print(" SIXTHSENSE OBJECT-DEPTH PRIORITIZATION")
print("========================================")

if not YOLO_PATH.exists():
    raise FileNotFoundError(
        f"YOLO model not found: {YOLO_PATH}"
    )

if not DEPTH_PATH.exists():
    raise FileNotFoundError(
        f"Depth model not found: {DEPTH_PATH}"
    )

torch.set_num_threads(2)

print("\nLoading YOLO...")
yolo_model = YOLO(str(YOLO_PATH))
print("YOLO loaded!")

print("\nLoading Depth Anything ONNX...")

options = ort.SessionOptions()

options.intra_op_num_threads = 2
options.inter_op_num_threads = 1

options.graph_optimization_level = (
    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
)

depth_session = ort.InferenceSession(
    str(DEPTH_PATH),
    sess_options=options,
    providers=["CPUExecutionProvider"]
)

print("Depth model loaded!")


# --------------------------------------------------
# 3. PREPROCESS IMAGE FOR DEPTH MODEL
# --------------------------------------------------

def preprocess_depth(frame):

    height, width = frame.shape[:2]

    scale = min(
        INPUT_SIZE / width,
        INPUT_SIZE / height
    )

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_CUBIC
    )

    rgb = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB
    )

    image = rgb.astype(np.float32) / 255.0

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    )

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

    input_array = np.transpose(
        padded,
        (2, 0, 1)
    )

    input_array = np.expand_dims(
        input_array,
        axis=0
    )

    padding_info = (
        pad_left,
        pad_top,
        new_width,
        new_height
    )

    return input_array, padding_info


# --------------------------------------------------
# 4. GENERATE DEPTH MAP
# --------------------------------------------------

def estimate_depth(frame):

    input_array, padding_info = preprocess_depth(
        frame
    )

    input_name = depth_session.get_inputs()[0].name
    output_name = depth_session.get_outputs()[0].name

    output = depth_session.run(
        [output_name],
        {input_name: input_array}
    )[0]

    depth = np.squeeze(output)

    pad_left, pad_top, new_width, new_height = (
        padding_info
    )

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
# 5. DETERMINE OBJECT DIRECTION
# --------------------------------------------------

def get_direction(center_x, frame_width):

    if center_x < frame_width / 3:
        return "LEFT"

    elif center_x < 2 * frame_width / 3:
        return "CENTRE"

    else:
        return "RIGHT"


# --------------------------------------------------
# 6. ASSOCIATE OBJECTS WITH DEPTH
# --------------------------------------------------

def associate_objects_with_depth(
    frame,
    depth_map,
    results
):

    height, width = frame.shape[:2]

    annotated_frame = frame.copy()
    obstacles = []

    # Direction boundaries
    first_boundary = width // 3
    second_boundary = 2 * width // 3

    cv2.line(
        annotated_frame,
        (first_boundary, 0),
        (first_boundary, height),
        (255, 255, 0),
        1
    )

    cv2.line(
        annotated_frame,
        (second_boundary, 0),
        (second_boundary, height),
        (255, 255, 0),
        1
    )

    boxes = results[0].boxes

    if boxes is None:
        return annotated_frame, obstacles

    for box in boxes:

        confidence = float(box.conf[0].item())

        # Ignore low-confidence detections
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
            .astype(int)
        )

        class_id = int(box.cls[0].item())

        object_name = yolo_model.names[class_id]

        # Keep coordinates inside the image
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))

        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            continue

        # Central 50% of bounding box
        box_width = x2 - x1
        box_height = y2 - y1

        crop_x1 = x1 + box_width // 4
        crop_x2 = x2 - box_width // 4

        crop_y1 = y1 + box_height // 4
        crop_y2 = y2 - box_height // 4

        # Extract depth from central region
        object_depth = depth_map[
            crop_y1:crop_y2,
            crop_x1:crop_x2
        ]

        valid_depth = object_depth[
            np.isfinite(object_depth)
        ]

        if valid_depth.size == 0:
            continue

        median_depth = float(
            np.median(valid_depth)
        )

        # Determine direction
        center_x = (x1 + x2) // 2

        direction = get_direction(
            center_x,
            width
        )

        # Store obstacle data
        obstacles.append({
            "name": object_name,
            "direction": direction,
            "relative_depth": median_depth,
            "confidence": confidence
        })

        # Draw object bounding box
        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Draw central depth sampling region
        cv2.rectangle(
            annotated_frame,
            (crop_x1, crop_y1),
            (crop_x2, crop_y2),
            (255, 0, 0),
            1
        )

        label = (
            f"{object_name} | {direction} | "
            f"Score: {median_depth:.2f}"
        )

        cv2.putText(
            annotated_frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    return annotated_frame, obstacles


# --------------------------------------------------
# 6A. REGION STATUS & DIRECTION GUIDANCE
# --------------------------------------------------

def get_region_guidance(obstacles):

    all_regions = [
        "LEFT",
        "CENTRE",
        "RIGHT"
    ]

    blocked_regions = set()

    # Identify regions containing CLOSE obstacles
    for obstacle in obstacles:

        if obstacle["distance_level"] == "CLOSE":

            direction = obstacle["direction"].upper()

            if direction in all_regions:
                blocked_regions.add(direction)

    # All three regions blocked
    if len(blocked_regions) == 3:

        return (
            "ALL REGIONS BLOCKED! "
            "Please wait for obstacles to clear."
        )

    # Some regions blocked
    elif blocked_regions:

        clear_regions = [
            region
            for region in all_regions
            if region not in blocked_regions
        ]

        blocked_text = " and ".join(
            region
            for region in all_regions
            if region in blocked_regions
        )

        clear_text = " or ".join(
            clear_regions
        )

        return (
            f"{blocked_text} BLOCKED. "
            f"Potentially clear region(s): "
            f"{clear_text}. "
            f"Check before moving."
        )

    # No CLOSE obstacle detected
    else:

        return (
            "No region has a CLOSE obstacle "
            "detection in this frame."
        )


# --------------------------------------------------
# 7. PRINT PRIORITIZED OBSTACLE INFORMATION
# --------------------------------------------------

def print_obstacle_results(
    obstacles,
    detection_time,
    processing_time
):

    print("\n")
    print("=" * 60)
    print("       SIXTHSENSE OBSTACLE ALERT")
    print("=" * 60)

    print(f"Detection Time : {detection_time}")

    print(
        f"Processing Time: "
        f"{processing_time:.3f} seconds"
    )

    if not obstacles:

        print("\nNo accepted obstacles detected.")

        print(
            "No primary or secondary alert "
            "for this frame."
        )

        print(
            "\nREGION STATUS\n"
            "No CLOSE obstacle detections."
        )

    else:

        print("\nPRIORITIZED OBSTACLES")
        print("-" * 60)

        # Print every accepted obstacle
        for obstacle in obstacles:

            rank = obstacle.get(
                "priority_rank",
                obstacle.get("priority")
            )

            if rank == 1:
                alert_type = "PRIMARY ALERT"
            else:
                alert_type = "SECONDARY ALERT"

            print(
                f"\nPriority Rank : {rank}"
            )

            print(
                f"Alert Type   : {alert_type}"
            )

            print(
                f"Object       : "
                f"{obstacle['name']}"
            )

            print(
                f"Direction    : "
                f"{obstacle['direction']}"
            )

            print(
                f"Relative Score: "
                f"{obstacle['relative_depth']:.3f}"
            )

            print(
                f"Distance Level: "
                f"{obstacle['distance_level']}"
            )

            print(
                f"Confidence   : "
                f"{obstacle['confidence']:.3f}"
            )

            print(
                f"Priority Score: "
                f"{obstacle['priority_score']}"
            )

            print(
                f"Alert        : "
                f"{obstacle['alert']}"
            )

            print("-" * 60)

        # ------------------------------------------
        # PRIMARY ALERT
        # ------------------------------------------

        primary_obstacle = obstacles[0]

        print("\nPRIMARY ALERT FOR THIS FRAME")

        print(
            f"Rank 1 Object: "
            f"{primary_obstacle['name']}"
        )

        print(
            f"Direction    : "
            f"{primary_obstacle['direction']}"
        )

        print(
            f"Priority Score: "
            f"{primary_obstacle['priority_score']}"
        )

        print(
            f"Alert        : "
            f"{primary_obstacle['alert']}"
        )

        # ------------------------------------------
        # REGION STATUS & DIRECTION GUIDANCE
        # ------------------------------------------

        print("\nREGION STATUS")

        region_guidance = get_region_guidance(
            obstacles
        )

        print(region_guidance)

    print("=" * 60)


# --------------------------------------------------
# 8. START WEBCAM
# --------------------------------------------------

print("\nConnecting webcam...")

camera = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

if not camera.isOpened():
    raise RuntimeError(
        "Could not access webcam."
    )

print("Webcam connected!")
print("Press Q to exit.")


# --------------------------------------------------
# 9. MAIN PROCESSING LOOP
# --------------------------------------------------

frame_count = 0
combined_view = None

try:

    while True:

        success, frame = camera.read()

        if not success:

            print("Could not capture frame.")
            break

        frame_count += 1

        if (
            frame_count - 1
        ) % PROCESS_EVERY_N_FRAMES == 0:

            # Start timing the complete processing cycle
            start_time = time.perf_counter()

            # Detection timestamp
            detection_time = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            # ------------------------------------------
            # YOLO OBJECT DETECTION
            # ------------------------------------------

            results = yolo_model.predict(
                frame,
                imgsz=416,
                conf=0.25,
                device="cpu",
                verbose=False
            )

            # Debug: raw YOLO detections
            print("\nRaw YOLO detections:")

            boxes = results[0].boxes

            if boxes is not None and len(boxes) > 0:

                for box in boxes:

                    class_id = int(
                        box.cls[0].item()
                    )

                    confidence = float(
                        box.conf[0].item()
                    )

                    object_name = yolo_model.names[
                        class_id
                    ]

                    print(
                        f"{object_name:15} "
                        f"Confidence: {confidence:.2f}"
                    )

            else:

                print("No objects detected.")

            # ------------------------------------------
            # DEPTH ESTIMATION
            # ------------------------------------------

            depth_map = estimate_depth(frame)

            # ------------------------------------------
            # OBJECT-DEPTH ASSOCIATION
            # ------------------------------------------

            annotated_frame, obstacles = (
                associate_objects_with_depth(
                    frame,
                    depth_map,
                    results
                )
            )

            # ------------------------------------------
            # OBSTACLE PRIORITIZATION
            # ------------------------------------------

            prioritized_obstacles = (
                prioritize_obstacles(obstacles)
            )

            # ------------------------------------------
            # PROCESSING TIME
            # ------------------------------------------

            processing_time = (
                time.perf_counter() - start_time
            )

            # ------------------------------------------
            # PRINT FINAL RESULTS
            # ------------------------------------------

            print_obstacle_results(
                prioritized_obstacles,
                detection_time,
                processing_time
            )

            # ------------------------------------------
            # DEPTH VISUALIZATION
            # ------------------------------------------

            depth_normalized = cv2.normalize(
                depth_map,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            )

            depth_normalized = (
                depth_normalized.astype(np.uint8)
            )

            depth_colored = cv2.applyColorMap(
                depth_normalized,
                cv2.COLORMAP_INFERNO
            )

            # ------------------------------------------
            # DISPLAY BOTH PANELS
            # ------------------------------------------

            combined_view = np.hstack(
                (
                    annotated_frame,
                    depth_colored
                )
            )

            cv2.putText(
                combined_view,
                f"Processing: {processing_time:.2f}s",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                combined_view,
                detection_time,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            print(
                f"\nTotal processing time: "
                f"{processing_time:.3f}s"
            )

            print("-" * 65)

        # ----------------------------------------------
        # SHOW CAMERA WINDOW
        # ----------------------------------------------

        if combined_view is not None:

            cv2.imshow(
                "SixthSense - Object Depth Prioritization",
                combined_view
            )

        else:

            cv2.imshow(
                "SixthSense - Starting...",
                frame
            )

        # Exit on Q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    camera.release()
    cv2.destroyAllWindows()

    print(
        "\nSixthSense object-depth prioritization "
        "test completed!"
    )
