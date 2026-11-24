
import cv2
import numpy as np
import torch
import onnxruntime as ort
import time

from pathlib import Path
from ultralytics import YOLO


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

YOLO_PATH = PROJECT_ROOT / "yolo11n.pt"
DEPTH_PATH = BASE_DIR / "depth" / "depth_anything_v2_vits_392.onnx"

INPUT_SIZE = 392
PROCESS_EVERY_N_FRAMES = 5

# Potential navigation obstacles recognised by COCO YOLO

NAVIGATION_OBSTACLE_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "train",
    "bench",
    "chair",
    "couch",
    "potted plant",
    "dining table",
    "suitcase",
    "backpack",
    "handbag",
    "umbrella",
    "skateboard"
}


# --------------------------------------------------
# 2. LOAD MODELS
# --------------------------------------------------

print("\n========================================")
print(" SIXTHSENSE OBJECT-DEPTH ASSOCIATION")
print("========================================")

if not YOLO_PATH.exists():
    raise FileNotFoundError(f"YOLO model not found: {YOLO_PATH}")

if not DEPTH_PATH.exists():
    raise FileNotFoundError(f"Depth model not found: {DEPTH_PATH}")

torch.set_num_threads(2)

print("\nLoading YOLO...")
yolo_model = YOLO(str(YOLO_PATH))
print("YOLO loaded!")

print("\nLoading Depth Anything ONNX...")

options = ort.SessionOptions()
options.intra_op_num_threads = 2
options.inter_op_num_threads = 1
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

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

    padding_info = (pad_left, pad_top, new_width, new_height)

    return input_array, padding_info


# --------------------------------------------------
# 4. GENERATE DEPTH MAP
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
# 6. CALCULATE DEPTH INSIDE EACH OBJECT BOX
# --------------------------------------------------



def associate_objects_with_depth(frame, depth_map, results):

    height, width = frame.shape[:2]
    annotated_frame = frame.copy()

    # Store structured information about detected objects.
    detected_objects = []

    # Draw boundaries between left, centre and right.
    first_boundary = width // 3
    second_boundary = 2 * width // 3

    cv2.line(annotated_frame, (first_boundary, 0),
             (first_boundary, height), (255, 255, 0), 1)

    cv2.line(annotated_frame, (second_boundary, 0),
             (second_boundary, height), (255, 255, 0), 1)

    boxes = results[0].boxes

    if boxes is None:
        return annotated_frame, detected_objects

    for box in boxes:

        confidence = float(box.conf[0].item())

        if confidence < 0.50:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        class_id = int(box.cls[0].item())
        object_name = yolo_model.names[class_id]

        # Keep coordinates inside the image.
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            continue

        # Select the central 50% of the bounding box.
        box_width = x2 - x1
        box_height = y2 - y1

        crop_x1 = x1 + box_width // 4
        crop_x2 = x2 - box_width // 4
        crop_y1 = y1 + box_height // 4
        crop_y2 = y2 - box_height // 4

        object_depth = depth_map[
            crop_y1:crop_y2,
            crop_x1:crop_x2
        ]

        valid_depth = object_depth[np.isfinite(object_depth)]

        if valid_depth.size == 0:
            continue

        median_depth = float(np.median(valid_depth))

        # Determine horizontal direction.
        center_x = (x1 + x2) // 2
        direction = get_direction(center_x, width)

        # Store structured detection information.
        detection = {
            "object": object_name,
            "direction": direction,
            "relative_depth": median_depth,
            "confidence": confidence,
            "bbox": (x1, y1, x2, y2)
        }

        detected_objects.append(detection)

        # Draw original bounding box.
        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Draw central sampling region.
        cv2.rectangle(
            annotated_frame,
            (crop_x1, crop_y1),
            (crop_x2, crop_y2),
            (255, 0, 0),
            1
        )

        label = (
            f"{object_name} | {direction} | "
            f"Depth: {median_depth:.2f}"
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

        print(
            f"Object: {object_name:15} "
            f"Direction: {direction:6} "
            f"Relative depth score: {median_depth:.3f} "
            f"Confidence: {confidence:.2f}"
        )

    return annotated_frame, detected_objects



# --------------------------------------------------
# 7. FILTER POTENTIAL NAVIGATION OBSTACLES
# --------------------------------------------------

def filter_navigation_obstacles(detected_objects):

    potential_obstacles = []

    for detection in detected_objects:

        object_name = detection["object"].lower()

        if object_name in NAVIGATION_OBSTACLE_CLASSES:

            potential_obstacles.append(detection)

    return potential_obstacles


# --------------------------------------------------
# 8. GROUP OBSTACLES BY DIRECTION
# --------------------------------------------------

def group_obstacles_by_direction(potential_obstacles):

    directional_obstacles = {
        "LEFT": [],
        "CENTRE": [],
        "RIGHT": []
    }

    for obstacle in potential_obstacles:

        direction = obstacle["direction"]

        if direction in directional_obstacles:

            directional_obstacles[direction].append(obstacle)

    return directional_obstacles

    # --------------------------------------------------

# --------------------------------------------------
# 9. DECIDE SUGGESTED MOVEMENT DIRECTION
# --------------------------------------------------

def decide_movement_direction(directional_obstacles):

    left_blocked = bool(directional_obstacles["LEFT"])
    centre_blocked = bool(directional_obstacles["CENTRE"])
    right_blocked = bool(directional_obstacles["RIGHT"])

    # No obstacle detected
    if not left_blocked and not centre_blocked and not right_blocked:
        return "NO OBSTACLE DETECTED"

    # Both side directions are blocked
    if left_blocked and right_blocked:
        return "STOP"

    # Only left side is blocked
    if left_blocked and not centre_blocked:
        return "MOVE RIGHT"

    # Only right side is blocked
    if right_blocked and not centre_blocked:
        return "MOVE LEFT"

    # Centre is blocked: check available sides
    if centre_blocked:

        if not right_blocked:
            return "MOVE RIGHT"

        if not left_blocked:
            return "MOVE LEFT"

        return "STOP"

def stabilize_movement_decision(
    current_decision,
    previous_decision,
    consecutive_count
):

    # STOP must be immediate
    if current_decision == "STOP":
        return "STOP", "STOP", 2

            # No obstacle detected should also be immediate
    if current_decision == "NO OBSTACLE DETECTED":
        return "NO OBSTACLE DETECTED", "NO OBSTACLE DETECTED", 0

    # Count consecutive identical decisions
    if current_decision == previous_decision:
        consecutive_count += 1
    else:
        consecutive_count = 1

    # Confirm after 2 consecutive frames
    if consecutive_count >= 2:
        return current_decision, current_decision, consecutive_count

    return "CHECKING...", current_decision, consecutive_count    


# --------------------------------------------------
# 10. START WEBCAM
# --------------------------------------------------

print("\nConnecting webcam...")

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    raise RuntimeError("Could not access webcam.")

print("Webcam connected!")
print("Press Q to exit.")


# --------------------------------------------------
# 11. MAIN PROCESSING LOOP
# --------------------------------------------------

frame_count = 0
combined_view = None

try:
    previous_decision = None
    consecutive_count = 0

    while True:

        success, frame = camera.read()

        if not success:
            print("Could not capture frame.")
            break

        frame_count += 1

        if (frame_count - 1) % PROCESS_EVERY_N_FRAMES == 0:

            start_time = time.time()

            # Run YOLO on the captured frame.
            results = yolo_model.predict(
                frame,
                imgsz=416,
                conf=0.25,
                device="cpu",
                verbose=False
            )

            # Generate depth map from the same frame.
            depth_map = estimate_depth(frame)

            # Associate each detected object with its depth.
            annotated_frame, detected_objects = associate_objects_with_depth(
                frame,
                depth_map,
                results
            )

            
            potential_obstacles = filter_navigation_obstacles(
                detected_objects
            )

            directional_obstacles = group_obstacles_by_direction(
                potential_obstacles
            )

            print("\nAll detected objects:")
            print(detected_objects)

            print("\nPotential navigation obstacles:")
            print(potential_obstacles)

            print("\nObstacles grouped by direction:")

            for direction, obstacles in directional_obstacles.items():
                print(f"{direction}: {obstacles}")

            current_decision = decide_movement_direction(
               directional_obstacles
            )

            movement_decision, previous_decision, consecutive_count = (
                stabilize_movement_decision(
                    current_decision,
                    previous_decision,
                    consecutive_count
                )
            )

            print("\nSuggested Movement:")
            print(movement_decision)

            # Prepare depth visualization.
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

            inference_time = time.time() - start_time

            # Display both panels.
            combined_view = np.hstack(
                (annotated_frame, depth_colored)
            )

            cv2.putText(
                combined_view,
                f"Processing time: {inference_time:.2f}s",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            print(
                f"\nTotal processing time: {inference_time:.2f}s"
            )
            print("-" * 65)

        if combined_view is not None:

            cv2.imshow(
                "SixthSense - Object Depth Association",
                combined_view
            )

        else:

            cv2.imshow("SixthSense - Starting...", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    camera.release()
    cv2.destroyAllWindows()

    print("\nObject-depth association test completed!")
