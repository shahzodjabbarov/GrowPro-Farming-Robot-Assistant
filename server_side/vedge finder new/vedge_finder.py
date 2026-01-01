'''
#
# THis code trains the YOLOv8s model with the multiplied dataset (in this case balanced_dataset_ssppss)
#
# IMPORTANT, if you wanna train a new model DONT just run it or it will overwrite the existing one.
#
# creates, yolo files, yaml, etc. needed for the model
# In case you wanna retrain the models (delete folds, delete yaml, delete yolov8s, delete prediction_ssppss)
# TIPP: dont retrain, just take this one, it is good
'''

import os
import shutil
import random
import cv2
from ultralytics import YOLO
from sklearn.model_selection import KFold

# === KONFIGURATION ===
DATASET_DIR = "balanced_dataset_ssppss"
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png']
NUM_FOLDS = 3
EPOCHS = 50
IMG_SIZE = 300
FINAL_EXPORT_NAME = "prediction_ssppss.pt"
CLASS_NAMES = [ "Pumpkin A" , "Pumpkin Bro" , "Salad A", "Salad Bro","Strawberries A", "Strawberries Bro" ]

# === FUNKTIONEN ===

def clean_and_create(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def polygon_to_yolo(line):
    coords = list(map(float, line.strip().split()))
    class_id = int(coords[0])
    xs = coords[1::2]
    ys = coords[2::2]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def visualize_labels(image_dir, label_dir, output_dir="label_viz"):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(image_dir):
        if not filename.endswith(".jpg"):
            continue
        img_path = os.path.join(image_dir, filename)
        lbl_path = os.path.join(label_dir, filename.replace(".jpg", ".txt"))

        img = cv2.imread(img_path)
        if img is None or not os.path.exists(lbl_path):
            continue

        h, w = img.shape[:2]
        with open(lbl_path, "r") as f:
            for line in f:
                cls, xc, yc, bw, bh = map(float, line.strip().split())
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)
                color = (0, 255, 0)
                if int(cls) == 1:
                    color = (0, 0, 255)
                elif int(cls) == 2:
                    color = (255, 0, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, str(int(cls)), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imwrite(os.path.join(output_dir, filename), img)

# --- GET ALL IMAGE FILENAMES ---
all_images = [f for f in os.listdir(os.path.join(DATASET_DIR, "images")) if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS]
all_images.sort()

label_root = os.path.join(DATASET_DIR, "labels")
for filename in os.listdir(label_root):
    if filename.endswith(".txt"):
        label_path = os.path.join(label_root, filename)
        with open(label_path, "r") as f:
            lines = f.readlines()
        if not lines:
            continue
        new_lines = []
        for line in lines:
            if len(line.split()) > 5:
                new_lines.append(polygon_to_yolo(line))
            elif len(line.split()) == 5:
                new_lines.append(line.strip())
        with open(label_path, "w") as f:
            for l in new_lines:
                f.write(l + "\n")

kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
best_model_path = ""

for fold, (train_idx, val_idx) in enumerate(kf.split(all_images)):
    print(f"\n=== Fold {fold+1}/{NUM_FOLDS} ===")

    fold_dir = os.path.join(DATASET_DIR, f"fold{fold+1}")
    images_train = os.path.join(fold_dir, "images/train")
    images_val = os.path.join(fold_dir, "images/val")
    labels_train = os.path.join(fold_dir, "labels/train")
    labels_val = os.path.join(fold_dir, "labels/val")

    for d in [images_train, images_val, labels_train, labels_val]:
        clean_and_create(d)

    for idx_list, img_dst, lbl_dst in [
        (train_idx, images_train, labels_train),
        (val_idx, images_val, labels_val)
    ]:
        for i in idx_list:
            img_name = all_images[i]
            lbl_name = os.path.splitext(img_name)[0] + ".txt"

            img_src_path = os.path.join(DATASET_DIR, "images", img_name)
            lbl_src_path = os.path.join(DATASET_DIR, "labels", lbl_name)

            if os.path.exists(img_src_path) and os.path.exists(lbl_src_path):
                shutil.copy(img_src_path, os.path.join(img_dst, img_name))
                shutil.copy(lbl_src_path, os.path.join(lbl_dst, lbl_name))

    print("\nChecking training labels...")
    empty_labels = 0
    for f in os.listdir(labels_train):
        with open(os.path.join(labels_train, f), "r") as file:
            lines = file.readlines()
            if not lines:
                print("⚠️ Empty label file:", f)
                empty_labels += 1
    print(f"Total empty labels in training set: {empty_labels}")

    visualize_labels(images_train, labels_train, output_dir=f"label_viz/fold{fold+1}_train")

    abs_path = os.path.abspath(fold_dir).replace("\\", "/")
    data_yaml = f"""
path: {abs_path}
train: images/train
val: images/val
nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""

    yaml_path = os.path.join(fold_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(data_yaml)

    print(f"✅ Fold {fold+1}: Dataset prepared with {len(train_idx)} train and {len(val_idx)} val images.")

    model = YOLO("yolov8s.pt")
    results = model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=4,
        lr0=0.01,
        optimizer="SGD",
        patience=40,
        degrees=30,
        translate=0.3,
        scale=0.8,
        shear=15,
        perspective=0.001,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.0,
        mixup=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        name=f"fold{fold+1}_train"
    )
    print(f"✅ Fold {fold+1} training complete.")

    if fold == NUM_FOLDS - 1:
        best_model_path = os.path.join("runs", "detect", f"fold{fold+1}_train", "weights", "best.pt")

if os.path.exists(best_model_path):
    shutil.copy(best_model_path, FINAL_EXPORT_NAME)
    print(f"✅ Final model exported as '{FINAL_EXPORT_NAME}'")
else:
    print("❌ Best model not found. Export failed.")




"""
import cv2
import numpy as np
import os


# --- Config ---
TEMPLATE_DIR = "templates"
TEST_IMAGE = "test_field.jpg"
MATCH_THRESHOLD = 0.5
MAX_SIZE = 100

# --- Helper: Remove white border from color image ---
def crop_white_border(img, threshold=245):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(255 - thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        return img[y:y+h, x:x+w]
    return img

# --- Helper: Non-Maximum Suppression ---
def non_max_suppression_fast(boxes, scores, overlap_thresh=0.4):
    if len(boxes) == 0:
        return [], []

    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= overlap_thresh)[0]
        order = order[inds + 1]

    return boxes[keep], scores[keep]

# --- Load templates (color-aware) ---
templates = []

for filename in os.listdir(TEMPLATE_DIR):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(TEMPLATE_DIR, filename)
        template = cv2.imread(path)
        if template is not None:
            template = crop_white_border(template)
            h, w = template.shape[:2]
            if h > MAX_SIZE or w > MAX_SIZE:
                scale = MAX_SIZE / max(h, w)
                template = cv2.resize(template, (int(w * scale), int(h * scale)))
                print(f"🔧 Resized + cropped template {filename} to {template.shape[1]}x{template.shape[0]}")
            hsv_template = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
            hue_template = hsv_template[:, :, 0]  # just hue channel
            templates.append((filename, hue_template))
        else:
            print(f"⚠️ Could not read {filename}")

if not templates:
    print("❌ No templates loaded.")
    exit()

print(f"✅ Loaded {len(templates)} hue templates.")

# --- Load test image ---
test_img = cv2.imread(TEST_IMAGE)
if test_img is None:
    print(f"❌ Could not load test image: {TEST_IMAGE}")
    exit()

test_img = cv2.resize(test_img, (800, 600))
hsv_test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2HSV)
hue_test_img = hsv_test_img[:, :, 0]  # use only hue channel for matching

# --- Run hue-based template matching ---
found_any = False

for filename, hue_template in templates:
    h, w = hue_template.shape
    res = cv2.matchTemplate(hue_test_img, hue_template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= MATCH_THRESHOLD)

    boxes = []
    scores = []

    for pt in zip(*loc[::-1]):
        boxes.append([pt[0], pt[1], w, h])
        scores.append(res[pt[1], pt[0]])

    if boxes:
        found_any = True
        print(f"🎯 {len(boxes)} hue matches found with template {filename}")
        nms_boxes, nms_scores = non_max_suppression_fast(boxes, scores, 0.4)

        for (x, y, w, h) in nms_boxes:
            cv2.rectangle(test_img, (x, y), (x + w, y + h), (0, 165, 255), 2)
            cv2.putText(test_img, "Pumpkin", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

if not found_any:
    print("🚫 No matches found.")

# --- Show final result ---
cv2.imshow("Pumpkin Hue Detection", test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""