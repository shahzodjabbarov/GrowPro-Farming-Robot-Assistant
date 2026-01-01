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

# Compute IoU between two boxes
def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

# Apply NMS
def apply_nms(predictions, iou_threshold=0.5):
    predictions = sorted(predictions, key=lambda x: x.confidence, reverse=True)
    keep = []

    for i, pred in enumerate(predictions):
        discard = False
        for kept in keep:
            iou = compute_iou(
                (pred.x - pred.width / 2, pred.y - pred.height / 2, pred.width, pred.height),
                (kept.x - kept.width / 2, kept.y - kept.height / 2, kept.width, kept.height)
            )
            if iou > iou_threshold:
                discard = True
                break
        if not discard:
            keep.append(pred)
    return keep

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Run inference
    results = model.infer(image=frame, confidence=0.75, iou_threshold=0.4)

    # Print and draw predictions
    print("\n--- Predictions ---")
    if results and results[0].predictions:
        raw_preds = results[0].predictions
        filtered_preds = apply_nms(raw_preds, iou_threshold=0.4)

        for i, pred in enumerate(filtered_preds):
            print(f"[{i}] Class: '{pred.class_name}' | Confidence: {pred.confidence:.2f} | Box: ({pred.x}, {pred.y}, {pred.width}, {pred.height})")

            x_center = int(pred.x)
            y_center = int(pred.y)
            width = int(pred.width)
            height = int(pred.height)

            x0 = x_center - width // 2
            y0 = y_center - height // 2
            x1 = x_center + width // 2
            y1 = y_center + height // 2

            # Color by class
            if pred.class_name.lower() == "qua binh thuong":
                color = (0, 255, 0)  # Green
            else:
                color = (0, 0, 255)  # Red

            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            label = f"{pred.class_name} ({pred.confidence:.2f})"
            cv2.putText(frame, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    else:
        print("No predictions.")

    # Show webcam feed
    cv2.imshow('Webcam Detection (NMS + Colored)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
