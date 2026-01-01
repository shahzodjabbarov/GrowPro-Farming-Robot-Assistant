from ultralytics import YOLO
import cv2
import gc

# 🧹 Clear memory in case an old model was loaded before
gc.collect()

# Load the trained model
model = YOLO("prediction_ssppss.pt")

# === DEBUG CHECK: print loaded class names ===
print("\n✅ Model classes loaded:")
for cls_id, cls_name in model.names.items():
    print(f"Class {cls_id}: {cls_name}")
print()

# Open camera (or change to 1 for USB camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Failed to open camera")
    exit()

print("🎥 Camera started... Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection with a confidence threshold
    results = model(frame, conf=0.65)
    annotated = results[0].plot()

    # Show the annotated frame
    cv2.imshow("Pumpkin Detection", annotated)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to break
        break

cap.release()
cv2.destroyAllWindows()
