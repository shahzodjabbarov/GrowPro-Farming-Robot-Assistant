import RPi.GPIO as GPIO
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from hardware import MotorControl
import socket
import struct
import pickle
import threading

# === Configuration ===
SERVER_IP = '192.168.230.89'
PORT = 8485

# === GPIO Setup ===
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

# === Ultrasonic Sensor Pins ===
TrigPin = 16
EchoPin = 18
GPIO.setup(TrigPin, GPIO.OUT)
GPIO.setup(EchoPin, GPIO.IN)

# === IR Sensor Pins ===
AvoidSensorLeft = 21
AvoidSensorRight = 19
Avoid_ON = 22
GPIO.setup(AvoidSensorLeft, GPIO.IN)
GPIO.setup(AvoidSensorRight, GPIO.IN)
GPIO.setup(Avoid_ON, GPIO.OUT)
GPIO.output(Avoid_ON, GPIO.HIGH)

# === Motor Init ===
motor_control = MotorControl()

# === Global variables for thread communication ===
streaming_active = True
navigation_active = True
camera1 = None
camera2 = None

# Navigation state tracking
navigation_state = {
    'last_direction': 'straight',
    'direction_confidence': 0,
    'ir_left_history': [],
    'ir_right_history': [],
    'stable_navigation': True
}

# === Ultrasonic Distance Function ===
def measure_distance():
    GPIO.output(TrigPin, GPIO.LOW)
    time.sleep(0.000002)
    GPIO.output(TrigPin, GPIO.HIGH)
    time.sleep(0.000015)
    GPIO.output(TrigPin, GPIO.LOW)
    start = time.time()
    while not GPIO.input(EchoPin):
        if time.time() - start > 0.03:
            return -1
    pulse_start = time.time()
    while GPIO.input(EchoPin):
        if time.time() - pulse_start > 0.03:
            return -1
    pulse_end = time.time()
    return ((pulse_end - pulse_start) * 340 / 2) * 100

# === Enhanced Lane Detection Functions ===
def preprocess_for_ground_detection(image):
    """
    Preprocess image for better ground-edge contrast detection
    """
    # Convert to grayscale for better edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(blurred)
    
    return enhanced

def create_ground_focused_roi(image):
    """
    Create ROI mask focusing on the ground area where the robot travels
    """
    h, w = image.shape
    
    # Create a trapezoidal ROI focusing on the ground ahead
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Define trapezoid points - focus on lower portion where ground is visible
    # Adjust these values based on your camera mounting angle
    top_y = int(h * 0.4)  # Start ROI from 40% down the image
    bottom_y = h
    
    # Trapezoid shape - wider at bottom, narrower at top
    left_top = int(w * 0.25)
    right_top = int(w * 0.75)
    left_bottom = int(w * 0.05)
    right_bottom = int(w * 0.95)
    
    roi_points = np.array([
        [left_bottom, bottom_y],
        [left_top, top_y],
        [right_top, top_y],
        [right_bottom, bottom_y]
    ], dtype=np.int32)
    
    cv2.fillPoly(roi_mask, [roi_points], 255)
    
    return roi_mask

def detect_ground_edges(image):
    """
    Detect edges between ground and side walls/barriers
    """
    # Preprocess image
    processed = preprocess_for_ground_detection(image)
    
    # Create ROI mask
    roi_mask = create_ground_focused_roi(processed)
    
    # Apply ROI mask
    roi_image = cv2.bitwise_and(processed, roi_mask)
    
    # Use adaptive edge detection
    # Canny edge detection with automatic threshold calculation
    med_val = np.median(roi_image[roi_image > 0])
    lower = int(max(0, 0.7 * med_val))
    upper = int(min(255, 1.3 * med_val))
    
    edges = cv2.Canny(roi_image, lower, upper, apertureSize=3, L2gradient=True)
    
    # Morphological operations to clean up edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    return edges, roi_mask

def detect_lane_boundaries(edges, original_frame):
    """
    Detect lane boundaries using Hough line detection
    """
    h, w = edges.shape
    
    # Hough line detection with parameters tuned for ground-edge detection
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=50,
        minLineLength=int(h * 0.15),  # Minimum line length relative to image height
        maxLineGap=20
    )
    
    left_lines = []
    right_lines = []
    line_image = original_frame.copy()
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate line angle and filter by angle
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            # Filter lines by angle (keep lines that could be lane boundaries)
            if abs(angle) > 15 and abs(angle) < 75:
                line_center_x = (x1 + x2) // 2
                
                # Classify as left or right line based on position
                if line_center_x < w // 2:
                    left_lines.append(line[0])
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green for left
                else:
                    right_lines.append(line[0])
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red for right
    
    return line_image, left_lines, right_lines

def calculate_navigation_direction(left_lines, right_lines, frame_width, ir_left, ir_right):
    """
    Calculate navigation direction using both visual and IR sensor data
    """
    global navigation_state
    
    # Update IR sensor history
    navigation_state['ir_left_history'].append(ir_left)
    navigation_state['ir_right_history'].append(ir_right)
    
    # Keep only last 5 readings
    if len(navigation_state['ir_left_history']) > 5:
        navigation_state['ir_left_history'].pop(0)
    if len(navigation_state['ir_right_history']) > 5:
        navigation_state['ir_right_history'].pop(0)
    
    # Calculate IR sensor trends
    ir_left_trend = sum(navigation_state['ir_left_history']) / len(navigation_state['ir_left_history'])
    ir_right_trend = sum(navigation_state['ir_right_history']) / len(navigation_state['ir_right_history'])
    
    direction = "straight"
    confidence = 0
    
    # Priority 1: IR sensors for immediate obstacle avoidance
    if ir_left == 0 and ir_right == 0:
        direction = "stop"
        confidence = 10
    elif ir_left == 0:
        direction = "right"
        confidence = 8
    elif ir_right == 0:
        direction = "left"
        confidence = 8
    else:
        # Priority 2: Visual lane detection
        left_detected = len(left_lines) > 0
        right_detected = len(right_lines) > 0
        
        if left_detected and right_detected:
            # Calculate lane center
            left_x = np.mean([line[0] + line[2] for line in left_lines]) / 2
            right_x = np.mean([line[0] + line[2] for line in right_lines]) / 2
            
            lane_center = (left_x + right_x) / 2
            frame_center = frame_width / 2
            
            offset = lane_center - frame_center
            threshold = frame_width * 0.05  # 5% of frame width
            
            if abs(offset) < threshold:
                direction = "straight"
                confidence = 7
            elif offset < 0:
                direction = "right"
                confidence = 6
            else:
                direction = "left"
                confidence = 6
                
        elif left_detected:
            direction = "right"  # Turn away from detected left boundary
            confidence = 4
        elif right_detected:
            direction = "left"   # Turn away from detected right boundary
            confidence = 4
        else:
            # No visual cues, use IR sensor trends
            if ir_left_trend < 0.5:  # Left side getting closer
                direction = "right"
                confidence = 3
            elif ir_right_trend < 0.5:  # Right side getting closer
                direction = "left"
                confidence = 3
            else:
                # Maintain last stable direction
                direction = navigation_state['last_direction']
                confidence = 1
    
    # Update navigation state
    if confidence > navigation_state['direction_confidence']:
        navigation_state['last_direction'] = direction
        navigation_state['direction_confidence'] = confidence
        navigation_state['stable_navigation'] = True
    else:
        # Decay confidence over time
        navigation_state['direction_confidence'] = max(0, navigation_state['direction_confidence'] - 1)
        if navigation_state['direction_confidence'] == 0:
            navigation_state['stable_navigation'] = False
    
    return direction, confidence

def execute_navigation_command(direction, confidence, motor_control):
    """
    Execute navigation command with smooth control
    """
    if direction == "stop":
        motor_control.stop()
    elif direction == "straight":
        motor_control.move_forward()
    elif direction == "left":
        if confidence > 6:
            motor_control.turn_left(slightly=False)  # Sharp turn for high confidence
        else:
            motor_control.turn_left(slightly=True)   # Gentle turn for low confidence
    elif direction == "right":
        if confidence > 6:
            motor_control.turn_right(slightly=False)  # Sharp turn for high confidence
        else:
            motor_control.turn_right(slightly=True)   # Gentle turn for low confidence

# === Initialize Cameras ===
def init_cameras():
    global camera1, camera2

    print("Opening first camera for lane detection...")
    try:
        camera1 = Picamera2()
        camera1.configure(camera1.create_preview_configuration(main={"format": 'YUV420', "size": (640, 480)}))
        camera1.start()
        time.sleep(2)
        print("First camera ready for lane detection.")
    except Exception as e:
        print(f"Error initializing first camera: {e}")
        camera1 = None

    print("Opening second camera for streaming...")
    try:
        camera2 = cv2.VideoCapture(1)
        if not camera2.isOpened():
            print("Second camera could not be opened.")
            camera2 = None
        else:
            print("Second camera ready for streaming.")
    except Exception as e:
        print(f"Error initializing second camera: {e}")
        camera2 = None

# === Camera Streaming Thread ===
def camera_streaming_thread():
    global streaming_active, camera2

    if camera2 is None:
        print("No second camera available for streaming.")
        return

    client_socket = None

    try:
        print(f"Connecting to server {SERVER_IP}:{PORT} ...")
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(20)
        client_socket.connect((SERVER_IP, PORT))
        print("✅ Streaming connection established.")

        while streaming_active:
            ret, frame = camera2.read()
            if not ret:
                print("No image from second camera.")
                break

            frame = cv2.resize(frame, (640, 480))
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
            success, buffer = cv2.imencode('.jpg', frame, encode_param)

            if not success:
                continue

            data = pickle.dumps(buffer)
            size = len(data)

            try:
                client_socket.sendall(struct.pack(">L", size) + data)
            except socket.error as e:
                print(f"❌ Streaming error: {e}")
                break

            time.sleep(0.03)

    except Exception as e:
        print(f"❌ Streaming thread error: {e}")
    finally:
        if client_socket:
            client_socket.close()
        print("🛑 Streaming connection closed.")

# === Main Navigation Thread ===
def navigation_thread():
    global navigation_active, camera1

    if camera1 is None:
        print("❌ No first camera available for navigation.")
        return

    loop_state = "outer_loop"
    turn_count = 0
    blink_counter = 0

    print("Starting outer loop navigation...")
    print("Navigation started. Press 'q' to stop.")

    while navigation_active:
        try:
            distance = measure_distance()
            left_ir = GPIO.input(AvoidSensorLeft)
            right_ir = GPIO.input(AvoidSensorRight)

            try:
                yuv_frame = camera1.capture_array("main")
                raw_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)
                
                # Enhanced lane detection
                edges, roi_mask = detect_ground_edges(raw_frame)
                processed_frame, left_lines, right_lines = detect_lane_boundaries(edges, raw_frame)
                
                # Calculate navigation direction with IR integration
                nav_direction, confidence = calculate_navigation_direction(
                    left_lines, right_lines, raw_frame.shape[1], left_ir, right_ir
                )

                # Visual feedback
                blink = (blink_counter // 10) % 2 == 0
                if left_ir == 0 and blink:
                    cv2.circle(processed_frame, (40, 40), 15, (0, 0, 255), -1)
                if right_ir == 0 and blink:
                    cv2.circle(processed_frame, (600, 40), 15, (0, 0, 255), -1)
                
                # Display navigation info
                cv2.putText(processed_frame, f"Dir: {nav_direction} (Conf: {confidence})", 
                           (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(processed_frame, f"IR L:{left_ir} R:{right_ir}", 
                           (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow("Raspbot Lane View", processed_frame)

            except Exception as e:
                print(f"Camera capture error: {e}")
                nav_direction = "straight"
                confidence = 1

            # Main navigation logic with enhanced lane detection
            if loop_state == "outer_loop":
                if distance != -1 and distance < 15:
                    motor_control.stop()
                    time.sleep(0.3)
                    motor_control.full_turn_right()
                    time.sleep(0.5)
                    turn_count += 1
                    if turn_count == 4:
                        motor_control.move_forward()
                        time.sleep(0.6)
                        motor_control.stop()
                        loop_state = "awaiting_inner_entry"
                        turn_count = 0
                        time.sleep(0.5)
                else:
                    execute_navigation_command(nav_direction, confidence, motor_control)

            elif loop_state == "awaiting_inner_entry":
                if distance != -1 and distance < 60:
                    motor_control.stop()
                    time.sleep(0.3)
                    motor_control.full_turn_right()
                    time.sleep(0.5)
                    loop_state = "inner_loop"
                    turn_count = 1

            elif loop_state == "inner_loop":
                if turn_count in [1, 2] and distance != -1 and distance < 10:
                    motor_control.stop()
                    time.sleep(0.3)
                    motor_control.full_turn_right()
                    time.sleep(0.5)
                    turn_count += 1
                    if turn_count == 3:
                        loop_state = "outer_loop"
                        turn_count = 0
                        time.sleep(1)
                else:
                    execute_navigation_command(nav_direction, confidence, motor_control)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("'q' pressed - stopping navigation...")
                break

            blink_counter += 1
            time.sleep(0.05)

        except Exception as e:
            print(f"Navigation error: {e}")
            time.sleep(0.1)

# === Main Program ===
def main():
    global streaming_active, navigation_active

    init_cameras()

    streaming_thread = threading.Thread(target=camera_streaming_thread, daemon=True)
    nav_thread = threading.Thread(target=navigation_thread, daemon=True)

    print("Starting streaming thread...")
    streaming_thread.start()

    print("Starting navigation thread...")
    nav_thread.start()

    try:
        while streaming_thread.is_alive() or nav_thread.is_alive():
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        print("Shutting down...")
        streaming_active = False
        navigation_active = False

        if streaming_thread.is_alive():
            streaming_thread.join(timeout=2)
        if nav_thread.is_alive():
            nav_thread.join(timeout=2)

        motor_control.stop()
        if camera1 is not None:
            camera1.stop()
        if camera2 is not None:
            camera2.release()
        GPIO.cleanup()
        cv2.destroyAllWindows()
        print("Cleanup complete.")

if __name__ == "__main__":
    main()