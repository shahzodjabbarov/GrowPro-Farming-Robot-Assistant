from ultralytics import YOLO
import cv2
import gc

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

# Define which classes belong to each category
category_classes = {
    1: [0, 1],  # Pumpkins: Pumpkin A, Pumpkin Bro
    2: [2, 3],  # Salads: Salad A, Salad Bro
    3: [4, 5]   # Strawberries: Strawberries A, Strawberries Bro
}

category_names = {
    1: "Pumpkins",
    2: "Salads", 
    3: "Strawberries"
}

print(f"🎯 Selected category: {category_names[choice]}")
print(f"   Detecting classes: {category_classes[choice]}")

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

    # Run detection with a confidence threshold (detects all classes)
    results = model(frame, conf=0.3)
    
    # Filter results to only show selected category
    filtered_boxes = []
    original_result = results[0]
    
    if original_result.boxes is not None:
        # Get the classes we want to display
        target_classes = category_classes[choice]
        
        # Filter boxes based on selected category
        for i, cls in enumerate(original_result.boxes.cls):
            if int(cls) in target_classes:
                filtered_boxes.append(i)
    
    # Create a copy of the frame to annotate
    annotated = frame.copy()
    
    # If we have filtered detections, draw them
    if filtered_boxes and original_result.boxes is not None:
        # Get filtered data
        filtered_boxes_tensor = original_result.boxes.xyxy[filtered_boxes]
        filtered_conf = original_result.boxes.conf[filtered_boxes]
        filtered_cls = original_result.boxes.cls[filtered_boxes]
        
        # Draw bounding boxes and labels
        for i, (box, conf, cls) in enumerate(zip(filtered_boxes_tensor, filtered_conf, filtered_cls)):
            x1, y1, x2, y2 = map(int, box)
            class_name = model.names[int(cls)]
            label = f"{class_name} {conf:.2f}"
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Add category info to the display
    info_text = f"Showing: {category_names[choice]} (Choice: {choice})"
    cv2.putText(annotated, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show the annotated frame
    cv2.imshow("Selective Detection", annotated)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to break
        break

cap.release()
cv2.destroyAllWindows()