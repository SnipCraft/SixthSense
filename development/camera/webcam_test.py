
import cv2

print("Starting SixthSense webcam test...")

# Try the default webcam
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("ERROR: Could not access webcam.")
    exit()

print("Webcam connected successfully!")
print("Press Q to close the camera.")

while True:
    success, frame = camera.read()

    if not success:
        print("ERROR: Failed to capture frame.")
        break

    cv2.imshow("SixthSense - Live Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("Webcam test completed.")
