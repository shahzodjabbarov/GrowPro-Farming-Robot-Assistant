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

# === Lane Detection Functions ===
def apply_green_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([70, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def perspective_roi_mask(mask):
    h, w = mask.shape
    top = int(h * 0.2)
    roi = np.zeros_like(mask)
    cv2.fillPoly(roi, [np.array([(0, h), (w, h), (int(w*0.85), top), (int(w*0.15), top)])], 255)
    cv2.fillPoly(roi, [np.array([(int(w*0.2), top), (int(w*0.8), top), (int(w*0.75), h), (int(w*0.25), h)])], 0)
    return cv2.bitwise_and(mask, roi)

def detect_lines(mask, frame):
    lines = cv2.HoughLinesP(mask, 2, np.pi / 180, 110, minLineLength=150, maxLineGap=15)
    line_positions = []
    line_image = frame.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (70, 200, 200), 5)
            line_positions.append(((x1 + x2) // 2, (y1 + y2) // 2))
    return line_image, line_positions

def navigate_in_lane(line_positions, frame_width):
    if len(line_positions) < 2:
        return "no_line"
    line_positions.sort(key=lambda p: p[0])
    left_line = line_positions[0]
    right_line = line_positions[-1]
    lane_center = (left_line[0] + right_line[0]) // 2
    frame_center = frame_width // 2
    if abs(lane_center - frame_center) < 20:
        return "straight"
    elif lane_center < frame_center:
        return "right"
    else:
        return "left"

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
                overlay = raw_frame.copy()

                green_mask = apply_green_mask(raw_frame)
                cropped_mask = perspective_roi_mask(green_mask)
                processed_frame, line_positions = detect_lines(cropped_mask, overlay)
                nav = navigate_in_lane(line_positions, raw_frame.shape[1])

                blink = (blink_counter // 10) % 2 == 0
                if left_ir == 0 and blink:
                    cv2.circle(processed_frame, (40, 40), 15, (0, 0, 255), -1)
                if right_ir == 0 and blink:
                    cv2.circle(processed_frame, (600, 40), 15, (0, 0, 255), -1)

                cv2.imshow("Raspbot Lane View", processed_frame)

            except Exception as e:
                print(f"Camera capture error: {e}")
                nav = "straight"
                processed_frame = None

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

                elif left_ir == 0 and right_ir == 0:
                    motor_control.stop()
                elif left_ir == 0:
                    motor_control.turn_right(slightly=True)
                elif right_ir == 0:
                    motor_control.turn_left(slightly=True)
                else:
                    if nav == "straight":
                        motor_control.move_forward()
                    elif nav == "left":
                        motor_control.turn_right(slightly=True)
                    elif nav == "right":
                        motor_control.turn_left(slightly=True)
                    else:
                        motor_control.move_forward()

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

                elif left_ir == 0 and right_ir == 0:
                    motor_control.stop()
                elif left_ir == 0:
                    motor_control.turn_right(slightly=True)
                elif right_ir == 0:
                    motor_control.turn_left(slightly=True)
                else:
                    if nav == "straight":
                        motor_control.move_forward()
                    elif nav == "left":
                        motor_control.turn_right(slightly=True)
                    elif nav == "right":
                        motor_control.turn_left(slightly=True)
                    else:
                        motor_control.move_forward()

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

