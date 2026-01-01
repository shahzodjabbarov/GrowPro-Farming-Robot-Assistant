import RPi.GPIO as GPIO
import time
from hardware import MotorControl  # Your custom motor control class

# Initialize motor controller
motor_control = MotorControl()

# GPIO setup
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

# Ultrasonic sensor pins
TrigPin = 16  # Adjust if different
EchoPin = 18

GPIO.setup(TrigPin, GPIO.OUT)
GPIO.setup(EchoPin, GPIO.IN)

# Function to measure distance
def measure_distance():
    GPIO.output(TrigPin, GPIO.LOW)
    time.sleep(0.000002)
    GPIO.output(TrigPin, GPIO.HIGH)
    time.sleep(0.000015)
    GPIO.output(TrigPin, GPIO.LOW)

    start_time = time.time()
    while not GPIO.input(EchoPin):
        if time.time() - start_time > 0.03:
            return -1
    pulse_start = time.time()

    while GPIO.input(EchoPin):
        if time.time() - pulse_start > 0.03:
            return -1
    pulse_end = time.time()

    distance_cm = ((pulse_end - pulse_start) * 340 / 2) * 100
    return distance_cm

# Main loop
try:
    while True:
        distance = measure_distance()

        if distance == -1:
            print("Measurement timeout")
            continue

        print(f"Distance: {distance:.2f} cm")

        if distance > 10:
            print("Clear path - moving forward")
            motor_control.move_forward()
        else:
            print("Obstacle too close - stopping")
            motor_control.stop()

        time.sleep(0.1)

except KeyboardInterrupt:
    print("Program interrupted by user")

finally:
    motor_control.stop()
    GPIO.cleanup()
    print("Program ended, GPIO cleaned")
