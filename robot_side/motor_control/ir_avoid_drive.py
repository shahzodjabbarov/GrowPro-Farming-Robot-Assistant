# -*- coding:UTF-8 -*-
import RPi.GPIO as GPIO
import time
from ir_sensing import Distance_test
from motor_control import MotorControl  # Assuming this is saved in motor_control.py

# === GPIO Setup ===
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

# === IR Sensor Pins ===
AvoidSensorLeft = 21
AvoidSensorRight = 19
Avoid_ON = 22

GPIO.setup(AvoidSensorLeft, GPIO.IN)
GPIO.setup(AvoidSensorRight, GPIO.IN)
GPIO.setup(Avoid_ON, GPIO.OUT)
GPIO.output(Avoid_ON, GPIO.HIGH)

# === Initialize Car ===
car = MotorControl()

print("🚗 Starting narrow path navigation...")

try:
    while True:
        # Check forward ultrasonic distance
        distance_ahead = Distance_test()
        left_sensor = GPIO.input(AvoidSensorLeft)
        right_sensor = GPIO.input(AvoidSensorRight)

        if distance_ahead != -1 and distance_ahead < 15:
            print("🔚 Obstacle ahead. Stopping.")
            car.stop()
            break

        # If wall on the right is too close, steer slightly left
        if right_sensor == 0 and distance_ahead > 15:
            print("↖️ Right wall <10 cm → steering slightly left")
            car.turn_left(slightly=True)
            time.sleep(0.5)
            car.stop()
        else:
            # Keep moving forward while checking side walls
            if left_sensor == 0 and right_sensor == 0:
                print("⬆️ Narrow path detected on both sides → move forward")
                car.move_forward()
            elif left_sensor == 0:
                print("↗️ Wall on LEFT → slight right")
                car.turn_right(slightly=True)
            elif right_sensor == 0:
                print("↖️ Wall on RIGHT → slight left")
                car.turn_left(slightly=True)
            else:
                print("🟡 No wall detected → move forward")
                car.move_forward()

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n🛑 Interrupted by user.")
    car.stop()

finally:
    GPIO.cleanup()
    print("✅ GPIO cleaned up.")
