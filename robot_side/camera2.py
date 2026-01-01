import cv2
from inference_sdk import InferenceHTTPClient

# Initialize Roboflow API client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="39xbdvLeHPcDz63z0ZTm"  # <- replace with your Roboflow API key
)

MODEL_ID = "planeacv/2"  # <- replace with your model ID (project/version)

# Start webcam
cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Save current frame temporarily
    cv2.imwrite("temp_frame.jpg", frame)

    # Run inference on the frame
    result = CLIENT.infer("temp_frame.jpg", model_id=MODEL_ID)

    # Draw the predictions
    for prediction in result["predictions"]:
        x, y, w, h = prediction["x"], prediction["y"], prediction["width"], prediction["height"]
        class_name = prediction["class"]
        confidence = prediction["confidence"]

        # Convert center x/y to top-left and bottom-right
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name} ({confidence:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Roboflow Webcam Inference", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
