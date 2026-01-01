import os
import shutil
import random
import cv2


'''
# Origin data test_dataset_ssppss  ---> Final output, balanced_dataset_ssppss
#
# This code just takes a folder of pictures and creates a dataset with a variety of pictures, tilted, angled, different lighting
# Purpose, training data creation
#
# Irrelevant, anlready trained model
'''



# === KONFIGURATION ===
# Original YOLO-Ordner
ORIGINAL_IMAGE_DIR = "test_dataset_ssppss/images"
ORIGINAL_LABEL_DIR = "test_dataset_ssppss/labels"

# Ziel: Balanced Dataset
OUTPUT_IMAGE_DIR = "balanced_dataset_ssppss/images"
OUTPUT_LABEL_DIR = "balanced_dataset_ssppss/labels"

# Maximale Bilder pro dominanter Klasse (z.B. Kürbis)
MAX_PER_CLASS = 100

# Klassen-Infos (müssen mit deinen Labels übereinstimmen)
CLASS_NAMES = ["Strawberries A", "Strawberries B" , "Pumpkin A" , "Pumpkin B" , "Salad A", "Salad B"]

# Welche Klassen sollen augmentiert werden
AUGMENT_CLASSES = [0, 1, 2, 3, 4, 5]  # Alle Klassen
AUGMENT_FACTOR = 5  # Wieviele Kopien pro Originalbild

# === FUNKTIONEN ===

def clean_and_create(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def get_class_distribution(label_dir):
    class_files = {i: [] for i in range(len(CLASS_NAMES))}
    for label_file in os.listdir(label_dir):
        label_path = os.path.join(label_dir, label_file)
        with open(label_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                cls = int(line.strip().split()[0])
                class_files[cls].append(label_file.replace(".txt", ""))
    return class_files

def augment_image(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    return cv2.warpAffine(img, M, (w, h))

def create_balanced_dataset():
    clean_and_create(OUTPUT_IMAGE_DIR)
    clean_and_create(OUTPUT_LABEL_DIR)

    class_files = get_class_distribution(ORIGINAL_LABEL_DIR)

    for cls, files in class_files.items():
        selected_files = files
        if len(files) > MAX_PER_CLASS and cls == 0:  # Nur Kürbis begrenzen
            selected_files = random.sample(files, MAX_PER_CLASS)

        print(f"Class {CLASS_NAMES[cls]}: {len(selected_files)} ausgewählte Bilder.")

        for file_stem in selected_files:
            img_src = os.path.join(ORIGINAL_IMAGE_DIR, file_stem + ".jpg")
            lbl_src = os.path.join(ORIGINAL_LABEL_DIR, file_stem + ".txt")

            img_dst = os.path.join(OUTPUT_IMAGE_DIR, file_stem + ".jpg")
            lbl_dst = os.path.join(OUTPUT_LABEL_DIR, file_stem + ".txt")

            if os.path.exists(img_src) and os.path.exists(lbl_src):
                shutil.copy(img_src, img_dst)
                shutil.copy(lbl_src, lbl_dst)

def augment_small_classes():
    for label_file in os.listdir(OUTPUT_LABEL_DIR):
        label_path = os.path.join(OUTPUT_LABEL_DIR, label_file)
        img_name = label_file.replace(".txt", ".jpg")
        img_path = os.path.join(OUTPUT_IMAGE_DIR, img_name)

        with open(label_path, "r") as f:
            classes = [int(line.split()[0]) for line in f.readlines()]

        if any(c in AUGMENT_CLASSES for c in classes):
            img = cv2.imread(img_path)
            if img is None:
                continue
            for i in range(AUGMENT_FACTOR):
                angle = random.randint(-20, 20)
                aug_img = augment_image(img, angle)

                aug_img_name = img_name.replace(".jpg", f"_aug{i}.jpg")
                aug_label_name = label_file.replace(".txt", f"_aug{i}.txt")

                cv2.imwrite(os.path.join(OUTPUT_IMAGE_DIR, aug_img_name), aug_img)
                shutil.copy(label_path, os.path.join(OUTPUT_LABEL_DIR, aug_label_name))

def create_data_yaml(output_dir):
    os.makedirs(output_dir, exist_ok=True)  # Ordner anlegen, falls er fehlt
    abs_path = os.path.abspath(output_dir).replace("\\", "/")

    with open(os.path.join(output_dir, "data.yaml"), "w") as f:
        f.write(f"path: {abs_path}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write("names:\n")
        for name in CLASS_NAMES:
            f.write(f"  - {name}\n")


# === PIPELINE AUSFÜHREN ===
if __name__ == "__main__":
    print("\n🚀 Starte Erstellung eines balancierten Datensatzes...")
    create_balanced_dataset()

    print("\n🎨 Starte Augmentation kleiner Klassen...")
    augment_small_classes()

    print("\n📄 Erstelle neue data.yaml...")
    create_data_yaml("balanced_dataset_ssppss")

    print("\n✅ Alles fertig! Benutze jetzt 'balanced_dataset_test/data.yaml' zum Trainieren.")
