import cv2
import numpy as np
import os

# Load in grayscale
image_path = 'C://Users//uaser//Desktop//b_w.jpg'
gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if gray is not None:
    # Invert the image
    inverted = 255 - gray

    # Create an alpha channel: 0 where original was white (background), 255 elsewhere
    alpha = np.where(gray == 255, 0, 255).astype(np.uint8)

    # Merge inverted (white lines) with alpha channel
    result = cv2.merge([inverted, inverted, inverted, alpha])  # 4-channel RGBA

    # Output folder and file
    output_dir = 'C://Users//uaser//Desktop//HISTORY'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, 'transparent_lines.png')
    success = cv2.imwrite(output_path, result)

    if success:
        print("Transparent image saved successfully.")
    else:
        print("Failed to save the image.")
else:
    print(f" Error: Couldn't load image from {image_path}")
