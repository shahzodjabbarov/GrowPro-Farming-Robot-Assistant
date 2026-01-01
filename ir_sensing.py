"""

# -*- coding:UTF-8 -*-
import RPi.GPIO as GPIO
import time
import YB_Pcb_Car

# === Setup ===
car = YB_Pcb_Car.YB_Pcb_Car()
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

# IR sensor pins are already defined in your original setup
AvoidSensorLeft = 21
AvoidSensorRight = 19
Avoid_ON = 22

# Ultrasonic pins (already initialized)
EchoPin = 18
TrigPin = 16

# Set up GPIO
GPIO.setup(AvoidSensorLeft, GPIO.IN)
GPIO.setup(AvoidSensorRight, GPIO.IN)
GPIO.setup(Avoid_ON, GPIO.OUT)
GPIO.output(Avoid_ON, GPIO.HIGH)
GPIO.setup(EchoPin, GPIO.IN)
GPIO.setup(TrigPin, GPIO.OUT)

# === Ultrasonic Distance Test Function (from your code) ===
def Distance_test():
    num = 0
    ultrasonic = []
    while num < 5:
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

        distance = ((pulse_end - pulse_start) * 340 / 2) * 100
        if 0 < distance < 500:
            ultrasonic.append(distance)
            num += 1

    distance_avg = sum(ultrasonic[1:4]) / 3
    return distance_avg

# === Lane following logic ===
def follow_between_walls():
    try:
        while True:
            distance = Distance_test()
            LeftSensorValue = GPIO.input(AvoidSensorLeft)
            RightSensorValue = GPIO.input(AvoidSensorRight)

            if distance != -1 and distance < 15:
                print("🔚 Reached end of lane. Stopping.")
                car.Car_Stop()
                break

            # Wall on the right side → turn slightly left
            if RightSensorValue == 0 and LeftSensorValue == 1:
                print("↖️ Wall on RIGHT → slight LEFT")
                car.Car_Run(40, 70)

            # Wall on the left side → turn slightly right
            elif LeftSensorValue == 0 and RightSensorValue == 1:
                print("↗️ Wall on LEFT → slight RIGHT")
                car.Car_Run(70, 40)

            # Both sides detected = center → go forward
            elif LeftSensorValue == 0 and RightSensorValue == 0:
                print("⬆️ Walls both sides → move forward")
                car.Car_Run(60, 60)

            # Neither side detected = open space → slow down or adjust
            elif LeftSensorValue == 1 and RightSensorValue == 1:
                print("🟡 No walls detected → slow search")
                car.Car_Run(40, 40)

            else:
                print("❓ Unknown state → stop")
                car.Car_Stop()

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("🛑 Interrupted by user")
        car.Car_Stop()

    finally:
        GPIO.cleanup()
        print("GPIO cleaned up.")

# === Start lane following ===
follow_between_walls()
"""
# -*- coding:UTF-8 -*-
import RPi.GPIO as GPIO
import time

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

print("📏 IR Proximity Test (~10 cm range). Press Ctrl+C to stop.\n")

try:
    while True:
        left = GPIO.input(AvoidSensorLeft)
        right = GPIO.input(AvoidSensorRight)

        if left == 0:
            print("👈 Obstacle detected by LEFT sensor (<~10 cm)")
        else:
            print("✅ LEFT sensor: no obstacle")

        if right == 0:
            print("👉 Obstacle detected by RIGHT sensor (<~10 cm)")
        else:
            print("✅ RIGHT sensor: no obstacle")

        time.sleep(0.3)

except KeyboardInterrupt:
    print("\n🛑 Test stopped by user.")

finally:
    GPIO.cleanup()
    print("✅ GPIO cleaned up.")
