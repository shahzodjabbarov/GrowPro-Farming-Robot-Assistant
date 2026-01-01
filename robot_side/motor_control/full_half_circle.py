import RPi.GPIO as GPIO
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from hardware import MotorControl

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

# === Motor and Camera Init ===
motor_control = MotorControl()
camera = Picamera2()
camera.configure(camera.create_preview_configuration(main={"format": 'YUV420', "size": (640, 480)}))
camera.start()
time.sleep(1)

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

# === Lane Detection ===
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
    top = int(h * 0.35)
    roi = np.zeros_like(mask)
    cv2.fillPoly(roi, [np.array([(0, h), (w, h), (int(w*0.95), top), (int(w*0.05), top)])], 255)
    cv2.fillPoly(roi, [np.array([(int(w*0.25), top), (int(w*0.75), top), (int(w*0.75), h), (int(w*0.25), h)])], 0)
    return cv2.bitwise_and(mask, roi)

def detect_lines(mask, frame):
    lines = cv2.HoughLinesP(mask, 2, np.pi / 180, 110, minLineLength=80, maxLineGap=30)
    line_positions = []
    line_image = frame.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (60, 200, 200), 5)
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

# === Navigation State Tracking ===
loop_state = "outer_loop"
turn_count = 0
print("🔁 Starting outer loop navigation...")

# === Main Control Loop ===
try:
    print("🤖 Navigation started. Press 'q' or Ctrl+C to stop.")
    blink_counter = 0

    while True:
        distance = measure_distance()
        left_ir = GPIO.input(AvoidSensorLeft)
        right_ir = GPIO.input(AvoidSensorRight)

        # === Camera Frame ===
        yuv_frame = camera.capture_array("main")
        raw_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)
        overlay = raw_frame.copy()

        # === Lane Detection ===
        green_mask = apply_green_mask(raw_frame)
        cropped_mask = perspective_roi_mask(green_mask)
        processed_frame, line_positions = detect_lines(cropped_mask, overlay)
        nav = navigate_in_lane(line_positions, raw_frame.shape[1])

        # === IR Blinking Indicators ===
        blink = (blink_counter // 10) % 2 == 0
        if left_ir == 0 and blink:
            cv2.circle(processed_frame, (40, 40), 15, (0, 0, 255), -1)
        if right_ir == 0 and blink:
            cv2.circle(processed_frame, (600, 40), 15, (0, 0, 255), -1)

        # === Navigation Logic ===
        if loop_state == "outer_loop":
            if distance != -1 and distance < 10:
                print(f"🔄 Outer loop turn {turn_count + 1}/4")
                motor_control.stop()
                time.sleep(0.3)
                motor_control.full_turn_left()
                time.sleep(0.5)
                turn_count += 1
                if turn_count == 4:
                    print("✅ Full outer circle complete! Moving forward a bit...")
                    motor_control.move_forward()
                    time.sleep(0.6)  # Adjust this based on your robot's speed
                    motor_control.stop()
                    loop_state = "awaiting_inner_entry"
                    turn_count = 0
                    time.sleep(0.5)


            elif left_ir == 0 and right_ir == 0:
                print("🛑 IR: Obstacles both sides - stopping")
                motor_control.stop()
            elif left_ir == 0:
                print("👈 IR: Obstacle left - slight right")
                motor_control.turn_right(slightly=True)
            elif right_ir == 0:
                print("👉 IR: Obstacle right - slight left")
                motor_control.turn_left(slightly=True)
            else:
                if nav == "straight":
                    motor_control.move_forward()
                elif nav == "left":
                    motor_control.turn_left(slightly=True)
                elif nav == "right":
                    motor_control.turn_right(slightly=True)
                else:
                    motor_control.move_forward()

        elif loop_state == "awaiting_inner_entry":
            if distance != -1 and distance < 60:
                print("🚪 Detected middle lane entry trigger")
                motor_control.stop()
                time.sleep(0.3)
                motor_control.full_turn_left()
                time.sleep(0.5)
                loop_state = "inner_loop"
                turn_count = 1  # Already did 1 wide turn

        elif loop_state == "inner_loop":
            if turn_count in [1, 2] and distance != -1 and distance < 10:
                print(f"🔄 Inner loop narrow turn {turn_count}/2")
                motor_control.stop()
                time.sleep(0.3)
                motor_control.full_turn_left()
                time.sleep(0.5)
                turn_count += 1
                if turn_count == 3:
                    print("✅ Half inner circle complete!")
                    loop_state = "outer_loop"
                    turn_count = 0
                    print("🔁 Restarting outer loop...")
                    time.sleep(1)

            elif left_ir == 0 and right_ir == 0:
                print("🛑 IR: Obstacles both sides - stopping")
                motor_control.stop()
            elif left_ir == 0:
                print("👈 IR: Obstacle left - slight right")
                motor_control.turn_right(slightly=True)
            elif right_ir == 0:
                print("👉 IR: Obstacle right - slight left")
                motor_control.turn_left(slightly=True)
            else:
                if nav == "straight":
                    motor_control.move_forward()
                elif nav == "left":
                    motor_control.turn_left(slightly=True)
                elif nav == "right":
                    motor_control.turn_right(slightly=True)
                else:
                    motor_control.move_forward()

        # === Show Debug Window ===
        cv2.imshow("📷 Raspbot Lane View", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        blink_counter += 1
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\n🛑 Interrupted by user.")

finally:
    motor_control.stop()
    camera.stop()
    GPIO.cleanup()
    cv2.destroyAllWindows()
