from picamera2 import Picamera2
import cv2
import numpy as np

## Lane Detection ###############################################
#   green masking
#   detecting lines 
#   (Changable is green filter, line length minimum, graph maximum
#################################################

def apply_green_mask(image):
    # Apply HSV mask for green areas
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Morphology to clean up noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

"""
def perspective_roi_mask(mask):
    # Crop to region of interest
    height, width = mask.shape
    top_crop = int(height * 0.55)
    bottom_left = (0, height)
    bottom_right = (width, height)
    top_left = (int(width * 0.10), top_crop)
    top_right = (int(width * 0.9), top_crop)
    polygon = np.array([[bottom_left, bottom_right, top_right, top_left]])
    
    
    
    roi_mask = np.zeros_like(mask)
    cv2.fillPoly(roi_mask, polygon, 255)
    return cv2.bitwise_and(mask, roi_mask)
  """  
    
def perspective_roi_mask(mask):
    height, width = mask.shape
    top_crop = int(height * 0.1)

    # Main trapezoid ROI
    bottom_left = (0, height)
    bottom_right = (width, height)
    top_left = (int(width * 0.05), top_crop)
    top_right = (int(width * 0.95), top_crop)
    outer_polygon = np.array([[bottom_left, bottom_right, top_right, top_left]])

    # Center square/rectangle cutout (20% width, full height)
    cutout_left = int(width * 0.25)
    cutout_right = int(width * 0.75
    )
    cutout_top = top_crop
    cutout_bottom = height
    inner_cutout = np.array([[
        (cutout_left, cutout_top),
        (cutout_right, cutout_top),
        (cutout_right, cutout_bottom),
        (cutout_left, cutout_bottom)
    ]])

    # Create mask and subtract the center cutout
    roi_mask = np.zeros_like(mask)
    cv2.fillPoly(roi_mask, outer_polygon, 255)      # Fill main ROI
    cv2.fillPoly(roi_mask, inner_cutout, 0)         # Subtract center area

    return cv2.bitwise_and(mask, roi_mask)



def detect_lines(mask, original_image):
    # Detect lines with Hough transform
    lines = cv2.HoughLinesP(mask, 2, np.pi / 180, 110, minLineLength=80, maxLineGap=30)
    line_image = np.copy(original_image)
    line_positions = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (60, 200, 200), 5)
            line_positions.append(((x1 + x2) // 2, (y1 + y2) // 2))  # Add midpoint of each line
    return line_image, line_positions


#################################################
# 
#
#################################################

def navigate_in_lane(line_positions, frame_width):
    # Find the leftmost and rightmost line midpoints
    if len(line_positions) < 2:
        # Not enough lines to navigate
        return "stop"
    
    # Sort lines by x-coordinate
    line_positions.sort(key=lambda p: p[0])
    left_line = line_positions[0]
    right_line = line_positions[-1]

    # Compute the lane center
    lane_center = (left_line[0] + right_line[0]) // 2
    frame_center = frame_width // 2

    # Determine direction based on lane center
    if abs(lane_center - frame_center) < 20:
        return "straight"
    elif lane_center < frame_center:
        return "right"
    else:
        return "left"








"""

import cv2
import numpy as np
#from hardware import display_camera_feed

def apply_green_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Morphology to clean noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def perspective_roi_mask(mask):
    height, width = mask.shape

    top_crop = int(height * 0.25)

    # Bottom edges (full width)
    bottom_left = (0, height)
    bottom_right = (width, height)

    # Top corners (narrower)
    top_left = (int(width * 0.20), top_crop)
    top_right = (int(width * 0.80), top_crop)

    # Define polygon points
    polygon = np.array([[bottom_left, bottom_right, top_right, top_left]])

    # Create mask and apply
    roi_mask = np.zeros_like(mask)
    cv2.fillPoly(roi_mask, polygon, 255)
    masked = cv2.bitwise_and(mask, roi_mask)
    return masked



def detect_lines(mask, original_image):
    lines = cv2.HoughLinesP(mask, 2, np.pi / 180, 110, minLineLength=300, maxLineGap=35)
    line_image = np.copy(original_image)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (100, 255, 255), 5)
    return line_image


# Load image
image = #cv2.imread("test_img.jpeg")
original = np.copy(image)

# Step 1: Green mask
green_mask = apply_green_mask(image)

# Step 2: Crop left/right 20%
cropped_mask = perspective_roi_mask(green_mask)

# Step 3: Detect and draw lane-style lines
lane_visual = detect_lines(cropped_mask, original)

# Show results
cv2.imshow("Binary Lettuce Mask", cropped_mask)
cv2.imshow("Lane Detection Result", lane_visual)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
