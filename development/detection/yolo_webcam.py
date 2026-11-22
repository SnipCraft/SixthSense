
import cv2
import torch
from ultralytics import YOLO

# Limit CPU threads to reduce system load
torch.set_num_threads(2)

print("Loading SixthSense YOLO model...")

# Load pretrained YOLO11 Nano model
model = YOLO("yolo11n.pt")

print("YOLO model loaded successfully!")

# Open laptop webcam
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("ERROR: Could not access webcam.")
    exit()

print("Camera connected!")
print("Press Q to quit.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Failed to capture frame.")
            break

        # Run object detection on CPU
        results = model.predict(
            frame,
            imgsz=416,
            conf=0.25,
            device="cpu",
            verbose=False

        )

        # Draw detection boxes and labels
        annotated_frame = results[0].plot()

        cv2.imshow(
            "SixthSense - YOLO Object Detection",
            annotated_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()

print("SixthSense YOLO test completed.")
