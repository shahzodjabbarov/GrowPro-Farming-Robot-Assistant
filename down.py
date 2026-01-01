from inference.models.utils import get_roboflow_model
import cv2

# Roboflow model
model_name = "planeacv"
model_version = "3"

# Open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Load Roboflow model
model = get_roboflow_model(
    model_id=f"{model_name}/{model_version}",
    api_key="39xbdvLeHPcDz63z0ZTm"
)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Run inference
    results = model.infer(image=frame, confidence=0.3, iou_threshold=0.4)

    # Print full result for debugging
    print("\n--- Predictions ---")
    if results and results[0].predictions:
        for i, pred in enumerate(results[0].predictions):
            print(f"[{i}] Class: '{pred.class_name}' | Confidence: {pred.confidence:.2f} | Box: ({pred.x}, {pred.y}, {pred.width}, {pred.height})")

            # Draw box
            x_center = int(pred.x)
            y_center = int(pred.y)
            width = int(pred.width)
            height = int(pred.height)

            x0 = x_center - width // 2
            y0 = y_center - height // 2
            x1 = x_center + width // 2
            y1 = y_center + height // 2

            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 2)
            label = f"{pred.class_name} ({pred.confidence:.2f})"
            cv2.putText(frame, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        print("No predictions.")

    # Show webcam feed
    cv2.imshow('Webcam Detection (Debug)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

