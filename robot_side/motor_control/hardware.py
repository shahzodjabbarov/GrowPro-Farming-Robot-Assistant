from picamera2 import Picamera2
import cv2
import sys
sys.path.append('/home/farmscout/Raspbot/2.Hardware Control course/02.Drive motor')
#sys.path.append("/home/farmscout/Raspbot/2.Hardware Control course/03.Drive servo")

from YB_Pcb_Car import YB_Pcb_Car


"""

#############  Both Cameras at once ###########

# Initialize PiCamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'YUV420'}))
picam2.start()

# Initialize USB camera
cap = cv2.VideoCapture(1)  # Adjust index to match your USB camera device

if not cap.isOpened():
    print("Error: Could not open USB camera.")
    exit()

try:
    while True:
        # Capture frame from PiCamera2
        yuv_frame = picam2.capture_array("main")
        pi_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)

        # Capture frame from USB camera
        ret, usb_frame = cap.read()
        if not ret:
            print("Error: Could not read USB camera frame.")
            break

        # Show both camera feeds in separate windows
        cv2.imshow("PiCamera Feed", pi_frame)
        cv2.imshow("USB Camera Feed", usb_frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Cleanup
    picam2.stop()
    cap.release()
    cv2.destroyAllWindows()
"""


"""
# Initialize and configure PiCamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'YUV420'}))
picam2.start()

def display_camera_feed():
    while True:
        # Capture frame in YUV format
        yuv_frame = picam2.capture_array("main")
        
        # Convert to BGR for OpenCV
        frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)

        # Show the camera feed directly
        cv2.imshow("Camera Feed", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Run it
display_camera_feed()

# Drive the car




# Use the second camera device
cap = cv2.VideoCapture(1)  # Change the index as needed

if not cap.isOpened():
    print("Error: Could not open USB camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    cv2.imshow("USB Camera Feed", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



"""

"""
####
class MotorControl:
    def __init__(self):
        self.car = YB_Pcb_Car()
        
    def move_forward(self):
        # Nutze die Methode aus YB_PcbMCar, um vorwärts zu fahren
        self.car.Car_Run(50, 50)  # Beispielgeschwindigkeit, ggf. anpassen
    
    def move_backward(self):
        # Nutze die Methode aus YB_PcbMCar, um vorwärts zu fahren
        self.car.Car_Back(50, 50)  # Beispielgeschwindigkeit, ggf. anpassen

    def turn_full(self):
        # Nutze die Methode aus YB_PcbMCar, um vorwärts zu fahren
        self.car.Car_Run(0, 50)  # Beispielgeschwindigkeit, ggf. anpassen
        self.car.Car_Back(50, 0)  # Beispielgeschwindigkeit, ggf. anpassen
 
    def turn_left(self, slightly=False):
        if slightly:
            self.car.Car_Left(40, 50)  # Leichte Linkskurve
        else:
            self.car.Car_Left(20, 50)  # Starke Linkskurve

    def turn_right(self, slightly=False):
        if slightly:
            self.car.Car_Right(50, 40)  # Leichte Rechtskurve
        else:
            self.car.Car_Right(50, 20)  # Starke Rechtskurve
    
    def turn_full_left(self):
        print("Turning full left (spin left)")
        self.car.Car_Spin_Left(50, 50)

    def turn_full_right(self):
        print("Turning full right (spin right)")
        self.car.Car_Spin_Right(50, 50)


    def stop(self):
        self.car.Car_Stop()
"""
class MotorControl:
    def __init__(self):
        self.car = YB_Pcb_Car()

    def move_forward(self):
        self.car.Car_Run(50, 50)  # Both motors forward at speed 50

    def move_backward(self):
        self.car.Car_Back(50, 50)  # Both motors backward at speed 50

    def turn_left(self, slightly=False):
        if slightly:
            print("Turning left slightly")
            self.car.Car_Run(30, 50)  # Left motor slower, right motor faster
        else:
            print("Turning left sharply")
            self.car.Car_Run(0, 50)  # Left motor stopped, right motor forward

    def turn_right(self, slightly=False):
        if slightly:
            print("Turning right slightly")
            self.car.Car_Run(50, 30)  # Right motor slower, left motor faster
        else:
            print("Turning right sharply")
            self.car.Car_Run(50, 0)  # Right motor stopped, left motor forward

    def stop(self):
        print("Stopping motors")
        self.car.Car_Stop()


