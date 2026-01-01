import cv2
import numpy as np

def analyze_lane_position(frame):
    """
    Analyzes the lane position and returns the direction and severity of off-track condition.
    
    Args:
        frame: The camera frame to analyze
        
    Returns:
        dict: Contains:
            - 'direction': 'center', 'left', or 'right' - the suggested correction direction
            - 'severity': int from 0-5, where 0 is centered and 5 is severely off-track
            - 'left_intensity': float representing the amount of lane markers on the left
            - 'right_intensity': float representing the amount of lane markers on the right
    """
    # Make a copy of the frame to avoid modifying the original
    processed_frame = frame.copy()
    
    # Get frame dimensions
    height, width = processed_frame.shape[:2]
    
    # Define regions of interest (ROI) for left and right sides
    left_roi = processed_frame[:, :width//3]
    center_roi = processed_frame[:, width//3:2*width//3]
    right_roi = processed_frame[:, 2*width//3:]
    
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2HSV)
    hsv_left = hsv[:, :width//3]
    hsv_right = hsv[:, 2*width//3:]
    
    # Define range for green color
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([90, 255, 255])
    
    # Create masks for green areas
    left_mask = cv2.inRange(hsv_left, lower_green, upper_green)
    right_mask = cv2.inRange(hsv_right, lower_green, upper_green)
    
    # Calculate the percentage of green pixels in each side
    left_green_percentage = np.sum(left_mask > 0) / (left_mask.shape[0] * left_mask.shape[1])
    right_green_percentage = np.sum(right_mask > 0) / (right_mask.shape[0] * right_mask.shape[1])
    
    # Calculate edge intensity for both sides as an alternative to color
    left_gray = cv2.cvtColor(left_roi, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_roi, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    left_edges = cv2.Canny(left_gray, 50, 150)
    right_edges = cv2.Canny(right_gray, 50, 150)
    
    # Calculate edge intensity (normalized sum of edge pixels)
    left_edge_intensity = np.sum(left_edges > 0) / (left_edges.shape[0] * left_edges.shape[1])
    right_edge_intensity = np.sum(right_edges > 0) / (right_edges.shape[0] * right_edges.shape[1])
    
    # Combine color and edge information for a more robust measure
    left_intensity = left_green_percentage * 0.7 + left_edge_intensity * 0.3
    right_intensity = right_green_percentage * 0.7 + right_edge_intensity * 0.3
    
    # Calculate intensity difference to determine position
    intensity_diff = left_intensity - right_intensity
    
    # Determine direction based on intensity difference
    if abs(intensity_diff) < 0.1:  # Threshold for being centered
        direction = "center"
        severity = 0
    elif intensity_diff > 0:  # More intensity on left suggests robot is too far right
        direction = "left"
        severity = min(5, max(1, int(abs(intensity_diff) * 10)))
    else:  # More intensity on right suggests robot is too far left
        direction = "right"
        severity = min(5, max(1, int(abs(intensity_diff) * 10)))
        
    # Create visualization for debugging
    visualization = processed_frame.copy()
    
    # Draw ROI boundaries
    cv2.line(visualization, (width//3, 0), (width//3, height), (0, 0, 255), 2)
    cv2.line(visualization, (2*width//3, 0), (2*width//3, height), (0, 0, 255), 2)
    
    # Add text with the results
    text = f"Direction: {direction}, Severity: {severity}"
    cv2.putText(visualization, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Add edge visualizations
    small_size = (width//4, height//4)
    left_edges_small = cv2.resize(left_edges, small_size)
    right_edges_small = cv2.resize(right_edges, small_size)
    
    # Convert edges to BGR
    left_edges_bgr = cv2.cvtColor(left_edges_small, cv2.COLOR_GRAY2BGR)
    right_edges_bgr = cv2.cvtColor(right_edges_small, cv2.COLOR_GRAY2BGR)
    
    # Place edge visualizations in corners
    visualization[10:10+small_size[1], 10:10+small_size[0]] = left_edges_bgr
    visualization[10:10+small_size[1], width-small_size[0]-10:width-10] = right_edges_bgr
    
    return {
        'direction': direction,
        'severity': severity,
        'left_intensity': float(left_intensity),
        'right_intensity': float(right_intensity),
        'visualization': visualization
    }

def integrate_with_lane_detection(frame):
    """
    A function that integrates with your existing lane detection code.
    It provides lane position analysis without controlling the robot.
    
    Args:
        frame: Camera frame
        
    Returns:
        tuple: (processed_frame, position_info)
            - processed_frame: Frame with visualization overlays
            - position_info: Dictionary with direction and severity
    """
    # Apply your existing lane detection preprocessing
    # (This would call your apply_green_mask, perspective_roi_mask, etc.)
    # For now, we'll use the lane position analyzer directly
    
    position_info = analyze_lane_position(frame)
    
    return position_info['visualization'], position_info

# Example usage (when running this file directly)
if __name__ == "__main__":
    # Example: Load a test image
    # image = cv2.imread("test_lane.jpg")
    
    # For webcam testing
    cap = cv2.VideoCapture(0)  # Use 0 for default camera
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Analyze lane position
        visualization, position_info = integrate_with_lane_detection(frame)
        
        # Display direction and severity
        print(f"Direction: {position_info['direction']}, Severity: {position_info['severity']}")
        
        # Show the visualization
        cv2.imshow("Lane Analysis", visualization)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()