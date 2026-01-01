import pygame
import sys
import os
import cv2
import numpy as np
import gc
import socket
import struct
import pickle
import threading
import time
from ultralytics import YOLO

# Setup
pygame.init()
WIDTH, HEIGHT = 955, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Manual Robot Control with Camera Feed and YOLO Detection")
clock = pygame.time.Clock()

# Robot control configuration
PI_IP = "192.168.238.173"  # Raspberry Pi's local IP
CONTROL_PORT = 8490
VIDEO_PORT = 8491

# Global variables
current_frame = None
frame_lock = threading.Lock()
video_socket_connected = False
control_socket_connected = False
connection_status = "Disconnected"

# Camera display area coordinates
CAMERA_TOP_LEFT = (559, 161)
CAMERA_BOTTOM_RIGHT = (888, 479)
CAMERA_WIDTH = CAMERA_BOTTOM_RIGHT[0] - CAMERA_TOP_LEFT[0]
CAMERA_HEIGHT = CAMERA_BOTTOM_RIGHT[1] - CAMERA_TOP_LEFT[1]

# YOLO Setup
print("Loading YOLO model...")
try:
    gc.collect()
    model = YOLO("prediction_ssppss.pt")
    choice = 3 # 1=Pumpkins, 2=Salads, 3=Strawberries
    category_classes = {1: [0, 1], 2: [2, 3], 3: [4, 5]}
    category_names = {1: "Pumpkins", 2: "Salads", 3: "Strawberries"}
    print(f"🎯 Selected category: {category_names[choice]}")
    print(f"   Detecting classes: {category_classes[choice]}")
    print("\n✅ Model classes loaded:")
    for cls_id, cls_name in model.names.items():
        print(f"Class {cls_id}: {cls_name}")
    yolo_enabled = True
except Exception as e:
    print(f"Warning: Could not load YOLO model: {e}")
    model = None
    yolo_enabled = False

class RobotController:
    def __init__(self):
        self.control_socket = None
        self.video_socket = None
        self.last_sent = None
        self.running = True
        self.key_map = {
            pygame.K_UP: 'UP', pygame.K_DOWN: 'DOWN', pygame.K_LEFT: 'LEFT',
            pygame.K_RIGHT: 'RIGHT', pygame.K_w: 'UP', pygame.K_s: 'DOWN',
            pygame.K_a: 'LEFT', pygame.K_d: 'RIGHT'
        }

    def connect_control(self):
        global control_socket_connected, connection_status
        for attempt in range(3):
            try:
                self.control_socket = socket.socket()
                self.control_socket.settimeout(5)
                self.control_socket.connect((PI_IP, CONTROL_PORT))
                control_socket_connected = True
                connection_status = f"Control Connected to {PI_IP}"
                print("[CONNECTED] Control to Raspberry Pi")
                return True
            except Exception as e:
                print(f"[ERROR] Control connection attempt {attempt + 1} failed: {e}")
                time.sleep(1)
        control_socket_connected = False
        connection_status = f"Control Error: Connection failed"
        return False

    def connect_video(self):
        global video_socket_connected, connection_status
        for attempt in range(3):
            try:
                self.video_socket = socket.socket()
                self.video_socket.settimeout(5)
                self.video_socket.connect((PI_IP, VIDEO_PORT))
                video_socket_connected = True
                connection_status = f"Video Connected to {PI_IP}"
                print("[CONNECTED] Video from Raspberry Pi")
                return True
            except Exception as e:
                print(f"[ERROR] Video connection attempt {attempt + 1} failed: {e}")
                time.sleep(1)
        video_socket_connected = False
        connection_status = f"Video Error: Connection failed"
        return False

    def send_command(self, command):
        global control_socket_connected
        if control_socket_connected and self.control_socket:
            try:
                if command != self.last_sent:
                    self.control_socket.sendall(command.encode())
                    self.last_sent = command
                    print(f"[SENT] {command}")
            except Exception as e:
                print(f"[ERROR] Failed to send command: {e}")
                control_socket_connected = False

    def send_stop(self):
        global control_socket_connected
        if control_socket_connected and self.control_socket:
            try:
                self.control_socket.sendall(b'STOP')
                self.last_sent = None
                print("[SENT] STOP")
            except Exception as e:
                print(f"[ERROR] Failed to send STOP: {e}")
                control_socket_connected = False

    def close(self):
        if self.control_socket:
            try:
                self.control_socket.sendall(b'QUIT')
                self.control_socket.close()
            except:
                pass
        if self.video_socket:
            try:
                self.video_socket.close()
            except:
                pass
        self.running = False

def video_receiver_thread(controller):
    global current_frame, video_socket_connected, connection_status
    if not controller.connect_video():
        video_socket_connected = False
        connection_status = "Video Connection Failed"
        return

    data = b""
    payload_size = struct.calcsize("!L")
    
    while controller.running and video_socket_connected:
        try:
            # Receive message size
            while len(data) < payload_size:
                packet = controller.video_socket.recv(4096)
                if not packet:
                    print("[INFO] Video connection closed by server")
                    video_socket_connected = False
                    break
                data += packet
            
            if not video_socket_connected:
                break

            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("!L", packed_msg_size)[0]

            # Receive frame data
            while len(data) < msg_size:
                packet = controller.video_socket.recv(min(4096, msg_size - len(data)))
                if not packet:
                    print("[INFO] Video connection closed during frame receive")
                    video_socket_connected = False
                    break
                data += packet
            
            if not video_socket_connected:
                break

            frame_data = data[:msg_size]
            data = data[msg_size:]

            # Decode frame
            frame_buffer = pickle.loads(frame_data)
            frame = cv2.imdecode(frame_buffer, cv2.IMREAD_COLOR)
            
            if frame is None:
                print("⚠️ Frame decode failed")
                continue

            with frame_lock:
                current_frame = frame.copy()

        except Exception as e:
            print(f"❌ Video processing error: {e}")
            video_socket_connected = False
            break

    video_socket_connected = False
    connection_status = "Video Disconnected"
    if controller.video_socket:
        try:
            controller.video_socket.close()
        except:
            pass
    print("✅ Video receiver ended")

# Initialize robot controller
robot_controller = RobotController()

# Start video receiver thread
video_thread = threading.Thread(target=video_receiver_thread, args=(robot_controller,), daemon=True)
video_thread.start()

# Connect control socket
robot_controller.connect_control()

# Load background and stickers
script_dir = os.path.dirname(os.path.abspath(__file__))
background = pygame.image.load(os.path.join(script_dir, "map.jpg")).convert()
sticker_img = pygame.image.load(os.path.join(script_dir, "car.png")).convert_alpha()
sticker_original = pygame.transform.scale(sticker_img, (sticker_img.get_width() // 7.5, sticker_img.get_height() // 7.5))

# Load vegetable stickers
bad_strawberry = pygame.image.load(os.path.join(script_dir, "bad_strawberry.png")).convert_alpha()
bad_pumpkin = pygame.image.load(os.path.join(script_dir, "bad_pumpkin.png")).convert_alpha()
bad_lettuce = pygame.image.load(os.path.join(script_dir, "bad_lettuce.png")).convert_alpha()

# Scale down vegetable stickers
scale_factor = 0.1
bad_strawberry_small = pygame.transform.scale(bad_strawberry, 
    (int(bad_strawberry.get_width() * scale_factor), int(bad_strawberry.get_height() * scale_factor)))
bad_pumpkin_small = pygame.transform.scale(bad_pumpkin, 
    (int(bad_pumpkin.get_width() * scale_factor), int(bad_pumpkin.get_height() * scale_factor)))
bad_lettuce_small = pygame.transform.scale(bad_lettuce, 
    (int(bad_lettuce.get_width() * scale_factor), int(bad_lettuce.get_height() * scale_factor)))

# Sticker mapping
placed_stickers = {}
STICKER_MAPPING = {
    pygame.K_1: (bad_strawberry_small, (66, 326)),
    pygame.K_2: (bad_strawberry_small, (65, 221)),
    pygame.K_3: (bad_pumpkin_small, (340, 185)),
    pygame.K_4: (bad_pumpkin_small, (400, 185)),
    pygame.K_5: (bad_lettuce_small, (490, 424)),
    pygame.K_6: (bad_lettuce_small, (445, 460)),
    pygame.K_7: (bad_lettuce_small, (167, 355)),
    pygame.K_8: (bad_strawberry_small, (192, 269))
}

# Road boundaries
ROADS = [
    (114, 215, 433, 245, 'horizontal'),
    (114, 300, 433, 330, 'horizontal'),
    (114, 388, 433, 418, 'horizontal'),
    (99, 230, 129, 403, 'vertical'),
    (418, 230, 448, 403, 'vertical'),
]

def is_on_road(x, y):
    for road in ROADS:
        x1, y1, x2, y2, _ = road
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False

def get_next_position(x, y, direction, speed):
    if direction == 'up':
        return x, y - speed
    elif direction == 'down':
        return x, y + speed
    elif direction == 'left':
        return x - speed, y
    elif direction == 'right':
        return x + speed, y
    return x, y

def get_direction_from_movement(old_x, old_y, new_x, new_y):
    dx = new_x - old_x
    dy = new_y - old_y
    if abs(dx) > abs(dy):
        return 'right' if dx > 0 else 'left'
    else:
        return 'down' if dy > 0 else 'up'

def get_rotated_sticker(direction):
    if direction == 'up':
        return sticker_original
    elif direction == 'left':
        return pygame.transform.rotate(sticker_original, 90)
    elif direction == 'down':
        return pygame.transform.rotate(sticker_original, 180)
    elif direction == 'right':
        return pygame.transform.rotate(sticker_original, -90)
    return sticker_original

def run_yolo_detection(frame):
    if not yolo_enabled or model is None:
        return frame
    try:
        results = model(frame, conf=0.4)
        filtered_boxes = []
        original_result = results[0]
        if original_result.boxes is not None:
            target_classes = category_classes[choice]
            for i, cls in enumerate(original_result.boxes.cls):
                if int(cls) in target_classes:
                    filtered_boxes.append(i)
        
        annotated = frame.copy()
        if filtered_boxes and original_result.boxes is not None:
            filtered_boxes_tensor = original_result.boxes.xyxy[filtered_boxes]
            filtered_conf = original_result.boxes.conf[filtered_boxes]
            filtered_cls = original_result.boxes.cls[filtered_boxes]
            for box, conf, cls in zip(filtered_boxes_tensor, filtered_conf, filtered_cls):
                x1, y1, x2, y2 = map(int, box)
                class_name = model.names[int(cls)]
                label = f"{class_name} {conf:.2f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(annotated, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        if yolo_enabled:
            info_text = f"Detecting: {category_names[choice]}"
            cv2.putText(annotated, info_text, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return annotated
    except Exception as e:
        print(f"YOLO detection error: {e}")
        return frame

def capture_camera_frame():
    global current_frame
    with frame_lock:
        if current_frame is None:
            return None
        frame = current_frame.copy()
    
    frame = run_yolo_detection(frame)
    frame_resized = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    try:
        frame_transposed = np.transpose(frame_rgb, (1, 0, 2))
        return pygame.surfarray.make_surface(frame_transposed)
    except Exception as e:
        print(f"Frame conversion failed: {e}")
        return None

# Initial state for visual car
x, y = 433, 403
speed = 0.8
direction = 'up'
keys_pressed = set()

# Main loop
running = True
print("🎮 Manual Robot Control started! Controls:")
print("   Arrow keys or WASD: Control robot movement")
print("   Number keys 1-8: Place vegetable stickers on map")
print("   C key: Clear all stickers")
print("   Q key: Change YOLO detection category")
print("   ESC: Exit")
print(f"\n📡 Connecting to robot at {PI_IP}...")

while running:
    screen.blit(background, (0, 0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key in robot_controller.key_map:
                keys_pressed.add(event.key)
                robot_controller.send_command(robot_controller.key_map[event.key])
            elif event.key in STICKER_MAPPING:
                placed_stickers[event.key] = STICKER_MAPPING[event.key]
            elif event.key == pygame.K_c:
                placed_stickers.clear()
            elif event.key == pygame.K_q and yolo_enabled:
                choice = (choice % 3) + 1
                print(f"🎯 Changed to category: {category_names[choice]}")
            elif event.key == pygame.K_ESCAPE:
                running = False
        elif event.type == pygame.KEYUP:
            if event.key in robot_controller.key_map:
                if event.key in keys_pressed:
                    keys_pressed.remove(event.key)
                if not any(key in robot_controller.key_map for key in keys_pressed):
                    robot_controller.send_stop()

    # Visual car movement
    keys = pygame.key.get_pressed()
    old_x, old_y = x, y
    dx, dy = 0, 0
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        dy -= speed
    if keys[pygame.K_DOWN] or keys[pygame.K_s]:
        dy += speed
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        dx -= speed
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        dx += speed
    
    if dx != 0 or dy != 0:
        new_x, new_y = x + dx, y + dy
        if is_on_road(new_x, new_y):
            x, y = new_x, new_y
            direction = get_direction_from_movement(old_x, old_y, x, y)

    # Draw stickers
    for key, (sticker_img, position) in placed_stickers.items():
        rect = sticker_img.get_rect(center=position)
        screen.blit(sticker_img, rect)

    # Draw visual car
    sticker = get_rotated_sticker(direction)
    rect = sticker.get_rect(center=(x, y))
    screen.blit(sticker, rect)
    
    # Draw camera feed
    camera_frame = capture_camera_frame()
    if camera_frame:
        screen.blit(camera_frame, CAMERA_TOP_LEFT)
    else:
        pygame.draw.rect(screen, (64, 64, 64), 
                        (CAMERA_TOP_LEFT[0], CAMERA_TOP_LEFT[1], CAMERA_WIDTH, CAMERA_HEIGHT))
        font = pygame.font.Font(None, 24)
        status_lines = ["Robot Camera", "Waiting for", "connection..."]
        for i, line in enumerate(status_lines):
            text = font.render(line, True, (255, 255, 255))
            text_rect = text.get_rect(center=(CAMERA_TOP_LEFT[0] + CAMERA_WIDTH//2, 
                                             CAMERA_TOP_LEFT[1] + CAMERA_HEIGHT//2 + (i-1)*30))
            screen.blit(text, text_rect)
    
    # Draw camera border
    border_color = (0, 255, 0) if video_socket_connected else (255, 255, 255)
    pygame.draw.rect(screen, border_color, 
                    (CAMERA_TOP_LEFT[0], CAMERA_TOP_LEFT[1], CAMERA_WIDTH, CAMERA_HEIGHT), 2)
    
    # Draw status
    font = pygame.font.Font(None, 24)
    video_color = (0, 255, 0) if video_socket_connected else (255, 0, 0)
    control_color = (0, 255, 0) if control_socket_connected else (255, 0, 0)
    video_text = f"Video: {'Connected' if video_socket_connected else 'Disconnected'}"
    control_text = f"Control: {'Connected' if control_socket_connected else 'Disconnected'}"
    screen.blit(font.render(video_text, True, video_color), (10, HEIGHT - 90))
    screen.blit(font.render(control_text, True, control_color), (10, HEIGHT - 60))
    
    if yolo_enabled:
        yolo_text = f"YOLO: {category_names[choice]} (Press Q to change)"
        yolo_color = (0, 255, 0)
    else:
        yolo_text = "YOLO: Disabled (Model not found)"
        yolo_color = (255, 0, 0)
    screen.blit(font.render(yolo_text, True, yolo_color), (10, HEIGHT - 30))

    pygame.display.flip()
    clock.tick(60)

# Cleanup
robot_controller.close()
pygame.quit()
sys.exit()