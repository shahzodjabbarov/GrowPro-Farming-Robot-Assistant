from ultralytics import YOLO
import cv2
import gc
import os

'''
Model classes
Class 0: Pumpkin A
Class 1: Pumpkin Bro
Class 2: Salad A
Class 3: Salad Bro
Class 4: Strawberries A
Class 5: Strawberries Bro
'''

# 🎯 CHOICE VARIABLE - Change this to select what to display
# 1 = Pumpkins (Class 0, 1)
# 2 = Salads (Class 2, 3)  
# 3 = Strawberries (Class 4, 5)
choice = 3

# === Class groupings ===
category_classes = {
    1: [0, 1],
    2: [2, 3],
    3: [4, 5]
}

category_names = {
    1: "Pumpkins",
    2: "Salads",
    3: "Strawberries"
}

# === File paths ===
model_path = "prediction_ssppss.pt"
image_path = "straw.jpg"  # Replace with your actual image path
output_path = "output_annotated.jpg"  # Will be saved in same folder

# === Load model ===
print(f"🎯 Selected category: {category_names[choice]}")
print(f"   Detecting classes: {category_classes[choice]}")
gc.collect()
model = YOLO(model_path)

# === Read image ===
frame = cv2.imread(image_path)
if frame is None:
    print(f"❌ Failed to load image: {image_path}")
    exit()

# === Run detection ===
results = model(frame, conf=0.3)
original_result = results[0]

# === Filter boxes by class ===
filtered_boxes = []
if original_result.boxes is not None:
    target_classes = category_classes[choice]
    for i, cls in enumerate(original_result.boxes.cls):
        if int(cls) in target_classes:
            filtered_boxes.append(i)

# === Draw annotations ===
annotated = frame.copy()
if filtered_boxes and original_result.boxes is not None:
    boxes = original_result.boxes.xyxy[filtered_boxes]
    confidences = original_result.boxes.conf[filtered_boxes]
    class_ids = original_result.boxes.cls[filtered_boxes]

    for box, conf, cls in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = map(int, box)
        class_name = model.names[int(cls)]
        label = f"{class_name} {conf:.2f}"

        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10),
                     (x1 + label_size[0], y1), (0, 255, 0), -1)

        # Draw text
        cv2.putText(annotated, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

# === Add overlay info ===
info_text = f"Showing: {category_names[choice]} (Classes: {category_classes[choice]})"
cv2.putText(annotated, info_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# === Show and Save ===
cv2.imshow("Selective Detection", annotated)
cv2.imwrite(output_path, annotated)
print(f"✅ Annotated image saved to: {os.path.abspath(output_path)}")

cv2.waitKey(0)
cv2.destroyAllWindows()
