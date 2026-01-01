
# Importiere die Module, die du bereits entwickelt hast
from hardware import MotorControl
#, display_camera_feed #check
from detection_lane_disease_vedgetable import *




#from Mode1 import PlantRecognition
#from Mode1 import DiseaseDetection
#from Mode1 import MoistureSensor
#from Mode1 import DataLogger


#from Mode2 import human_following
#from Mode1

#from picamera2 import Picamera2


from picamera2 import Picamera2

print(Picamera2.global_camera_info())





def drive_car_field_navigation():

     # Initialize camera and motor control
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": 'YUV420'}))
    picam2.start()
    motor_control = MotorControl()
    
    while True:
        # Capture a frame
        yuv_frame = picam2.capture_array("main")
        raw_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)
        
        # Process the frame
        green_mask = apply_green_mask(raw_frame)
        cropped_mask = perspective_roi_mask(green_mask)
        processed_frame, line_positions = detect_lines(cropped_mask, raw_frame)

        # Determine navigation
        command = navigate_in_lane(line_positions, raw_frame.shape[1])

        # Send commands to the car
        if command == "straight":
            motor_control.move_forward()
        elif command == "left":
            motor_control.turn_left()
        elif command == "right":
            motor_control.turn_right()
        else:
            motor_control.stop()

        # Display the processed frame with detected lines
        cv2.imshow("Lane Detection", processed_frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    picam2.stop()
    cv2.destroyAllWindows()

# Run the function
#drive_car_field_navigation()


import time

def drive_car_field_navigation_with_tracking():
    
    # Initialize camera and motor control
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": 'YUV420'}))
    picam2.start()
    motor_control = MotorControl()
    """
    cap = cv2.VideoCapture(1)  # Change to 0 if needed
    if not cap.isOpened():
        print("Error: Could not open USB camera.")
        return

    motor_control = MotorControl()
    """
    ######
    
    
    # Initialize tracking variables
    speed_m_per_s = 0.01  # Example speed in meters per second
    total_distance = 0.0
    last_printed_meter = 0
    last_time = time.time()

    while True:
        # Capture a frame
        """
        ret, raw_frame = cap.read()
        
        """
        yuv_frame = picam2.capture_array("main")
        raw_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)
        
        
        # Process the frame
        green_mask = apply_green_mask(raw_frame)
        cropped_mask = perspective_roi_mask(green_mask)
        processed_frame, line_positions = detect_lines(cropped_mask, raw_frame)

        # Determine navigation
        command = navigate_in_lane(line_positions, raw_frame.shape[1])

        # Get current time
        current_time = time.time()
        elapsed_time = current_time - last_time

        # Update distance based on speed and elapsed time
        if command in ["straight", "left", "right"]:
            total_distance += speed_m_per_s * elapsed_time

        # Print distance if it crosses a whole meter
        if int(total_distance) > last_printed_meter:
            last_printed_meter = int(total_distance)
            print(f"Total distance: {last_printed_meter} meters")

        last_time = current_time

        # Send commands to the car
        if command == "straight":
            motor_control.move_forward()
        elif command == "left":
            motor_control.turn_left(slightly=True)
        elif command == "right":
            motor_control.turn_right(slightly=True)
        else:
            motor_control.stop()

        # Display the processed frame with detected lines
        cv2.imshow("Lane Detection", processed_frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    picam2.stop()
    cv2.destroyAllWindows()

# Run the function
drive_car_field_navigation_with_tracking()


"""

class Raspbot:
    FIELD_SIZE = 6  # 6x6 grid (3m / 0.5m)
    
    def __init__(self):
        # Initialisiere alle Komponenten
        self.motor_control = MotorControl()
        self.camera = Camera()
        self.plant_recognition = PlantRecognition()
        self.disease_detection = DiseaseDetection()
        self.moisture_sensor = MoistureSensor()
        self.data_logger = DataLogger()
        self.field_navigation = FieldNavigation()
        
        # Aktueller Modus des Raspbots
        self.current_mode = 1  # z.B. Modus 1: Navigation und Pflanzenüberwachung
        
        # Mapping für das Feld: 6x6-Raster (3x3 Meter mit 50x50cm Zellen)
        self.field_map = [[None for _ in range(self.FIELD_SIZE)] for _ in range(self.FIELD_SIZE)]
        self.robot_position = [0, 0]  # Start in Zelle (0,0)
        
    def move_robot(self, direction):
        #Bewege den Roboter im Raster
        x, y = self.robot_position
        if direction == "up" and y > 0:
            self.robot_position[1] -= 1
        elif direction == "down" and y < self.FIELD_SIZE - 1:
            self.robot_position[1] += 1
        elif direction == "left" and x > 0:
            self.robot_position[0] -= 1
        elif direction == "right" and x < self.FIELD_SIZE - 1:
            self.robot_position[0] += 1

    def record_problem(self, problem_type):
        #Speichert ein Problem an der aktuellen Position.
        x, y = self.robot_position
        if self.field_map[y][x] is None:
            self.field_map[y][x] = [problem_type]
        else:
            self.field_map[y][x].append(problem_type)

    def display_map(self):
        #Zeigt das aktuelle Mapping des Feldes an.
        print("Feldmapping (X = Problem erkannt):")
        for row in self.field_map:
            row_str = ""
            for cell in row:
                if cell is None:
                    row_str += "[   ] "
                else:
                    row_str += "[ X ] "
            print(row_str)
        print("\n")

    def mode_1(self):
        #Implements Mode 1: Autonomous field navigation and plant monitoring
        print("Raspbot: Starte Navigation und Pflanzenüberwachung...")
        
        # Beispielablauf:
        # 1. Starte die Navigation zwischen den Reihen
        self.field_navigation.navigate()
        
        # 2. Erfasse Bilddaten mit der Kamera
        frame = self.camera.capture_frame()
        
        # 3. Erkenne Pflanzen im Bild
        plants = self.plant_recognition.find_plants(frame)
        
        for plant in plants:
            # 4. Überprüfe jede Pflanze auf Krankheiten
            if self.disease_detection.check_plant(plant):
                print(f"Erkannte kranke Pflanze: {plant.id}")
                
                # Speichere die Krankheit im Mapping
                self.record_problem("krankheit")
                
                # 5. Aktiviere den Motor für den Feuchtigkeitssensor
                print(f"Überprüfe Feuchtigkeit für Pflanze {plant.id}...")
                moisture_level = self.moisture_sensor.check_soil(plant.location)
                
                # 6. Protokolliere Daten in der Datenbank
                self.data_logger.log_disease(plant.id, moisture_level)
                
                # Falls Feuchtigkeit niedrig ist, speichere das im Mapping
                if moisture_level < self.moisture_sensor.THRESHOLD:
                    print(f"Warnung: Feuchtigkeit für Pflanze {plant.id} ist zu niedrig!")
                    self.data_logger.log_low_moisture(plant.id, moisture_level)
                    self.record_problem("trockenheit")

        # Zeige das Feldmapping nach dem Lauf
        self.display_map()

    def run(self):
        "#Hauptschleife, die den ausgewählten Modus ausführt
        if self.current_mode == 1:
            self.mode_1()
        else:
            print("Unbekannter Modus.")

if __name__ == "__main__":
    # Erzeuge das Raspbot-Objekt und starte die Hauptschleife
    raspbot = Raspbot()
    raspbot.run()
"""
