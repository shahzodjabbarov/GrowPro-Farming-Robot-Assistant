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
from collections import deque
import queue
from ultralytics import YOLO

# Setup
pygame.init()
WIDTH, HEIGHT = 955, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Manual Robot Control with Camera Feed and YOLO Detection")
clock = pygame.time.Clock()

# Robot control configuration
PI_IP = "192.168.238.173"   # Raspberry Pi's local IP
CONTROL_PORT = 8490
VIDEO_PORT = 8491

# Performance optimization settings
MAX_FPS = 30  # Reduced from 60 for better performance
YOLO_SKIP_FRAMES = 3  # Process YOLO every 3rd frame
CAMERA_RESIZE_FACTOR = 0.8  # Slightly reduce camera feed size for performance

# Global variables
frame_queue = queue.Queue(maxsize=2)  # Limit queue size to prevent memory buildup
processed_frame_cache = None
video_socket_connected = False
control_socket_connected = False
connection_status = "Disconnected"
yolo_frame_counter = 0

# Camera display area coordinates (slightly smaller for performance)
CAMERA_TOP_LEFT = (559, 161)
CAMERA_BOTTOM_RIGHT = (888, 479)
CAMERA_WIDTH = int((CAMERA_BOTTOM_RIGHT[0] - CAMERA_TOP_LEFT[0]) * CAMERA_RESIZE_FACTOR)
CAMERA_HEIGHT = int((CAMERA_BOTTOM_RIGHT[1] - CAMERA_TOP_LEFT[1]) * CAMERA_RESIZE_FACTOR)

# Pre-load and cache images for better performance
print("Loading assets...")
script_dir = os.path.dirname(os.path.abspath(__file__))

# Cache for pygame surfaces to avoid repeated conversions
surface_cache = {}

def load_and_cache_image(filename, scale_factor=1.0, convert_alpha=True):
    """Load and cache images to avoid repeated file I/O"""
    cache_key = f"{filename}_{scale_factor}_{convert_alpha}"
    if cache_key not in surface_cache:
        try:
            img = pygame.image.load(os.path.join(script_dir, filename))
            if convert_alpha:
                img = img.convert_alpha()
            else:
                img = img.convert()
            
            if scale_factor != 1.0:
                new_w = int(img.get_width() * scale_factor)
                new_h = int(img.get_height() * scale_factor)
                img = pygame.transform.scale(img, (new_w, new_h))
            
            surface_cache[cache_key] = img
        except Exception as e:
            print(f"Warning: Could not load {filename}: {e}")
            surface_cache[cache_key] = None
    
    return surface_cache[cache_key]

# Load all assets with caching
background = load_and_cache_image("map.jpg", convert_alpha=False)
sticker_original = load_and_cache_image("car.png", scale_factor=1/7.5)

# Load and scale vegetable stickers
scale_factor = 0.1
bad_strawberry_small = load_and_cache_image("bad_strawberry.png", scale_factor)
bad_pumpkin_small = load_and_cache_image("bad_pumpkin.png", scale_factor)
bad_lettuce_small = load_and_cache_image("bad_lettuce.png", scale_factor)

# Pre-calculate rotated car stickers for better performance
rotated_stickers = {}
if sticker_original:
    rotated_stickers['up'] = sticker_original
    rotated_stickers['left'] = pygame.transform.rotate(sticker_original, 90)
    rotated_stickers['down'] = pygame.transform.rotate(sticker_original, 180)
    rotated_stickers['right'] = pygame.transform.rotate(sticker_original, -90)

# YOLO Setup with optimization
print("Loading YOLO model...")
try:
    gc.collect()
    model = YOLO("prediction_ssppss.pt")
    # Optimize YOLO settings for performance
    model.overrides['verbose'] = False  # Reduce logging
    
    choice = 2  # 1=Pumpkins, 2=Salads, 3=Strawberries
    category_classes = {1: [0, 1], 2: [2, 3], 3: [4, 5]}
    category_names = {1: "Pumpkins", 2: "Salads", 3: "Strawberries"}
    print(f"🎯 Selected category: {category_names[choice]}")
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
        self.command_queue = deque(maxlen=5)  # Limit command queue
        self.key_map = {
            pygame.K_UP: 'UP', pygame.K_DOWN: 'DOWN', pygame.K_LEFT: 'LEFT',
            pygame.K_RIGHT: 'RIGHT', pygame.K_w: 'UP', pygame.K_s: 'DOWN',
            pygame.K_a: 'LEFT', pygame.K_d: 'RIGHT'
        }

    def connect_control(self):
        global control_socket_connected, connection_status
        try:
            self.control_socket = socket.socket()
            self.control_socket.settimeout(3)  # Reduced timeout
            self.control_socket.connect((PI_IP, CONTROL_PORT))
            control_socket_connected = True
            connection_status = f"Control Connected to {PI_IP}"
            print("[CONNECTED] Control to Raspberry Pi")
            return True
        except Exception as e:
            print(f"[ERROR] Control connection failed: {e}")
            control_socket_connected = False
            connection_status = f"Control Error: Connection failed"
            return False

    def connect_video(self):
        global video_socket_connected, connection_status
        try:
            self.video_socket = socket.socket()
            self.video_socket.settimeout(3)  # Reduced timeout
            self.video_socket.connect((PI_IP, VIDEO_PORT))
            video_socket_connected = True
            connection_status = f"Video Connected to {PI_IP}"
            print("[CONNECTED] Video from Raspberry Pi")
            return True
        except Exception as e:
            print(f"[ERROR] Video connection failed: {e}")
            video_socket_connected = False
            connection_status = f"Video Error: Connection failed"
            return False

    def send_command(self, command):
        global control_socket_connected
        if control_socket_connected and self.control_socket and command != self.last_sent:
            try:
                self.control_socket.sendall(command.encode())
                self.last_sent = command
            except Exception as e:
                print(f"[ERROR] Failed to send command: {e}")
                control_socket_connected = False

    def send_stop(self):
        global control_socket_connected
        if control_socket_connected and self.control_socket:
            try:
                self.control_socket.sendall(b'STOP')
                self.last_sent = None
            except Exception as e:
                control_socket_connected = False

    def close(self):
        self.running = False
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

def video_receiver_thread(controller):
    """Optimized video receiver with frame dropping"""
    global video_socket_connected, connection_status
    
    if not controller.connect_video():
        return

    data = b""
    payload_size = struct.calcsize("!L")
    frame_count = 0
    
    while controller.running and video_socket_connected:
        try:
            # Receive message size
            while len(data) < payload_size:
                packet = controller.video_socket.recv(8192)  # Increased buffer
                if not packet:
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
                remaining = msg_size - len(data)
                packet = controller.video_socket.recv(min(8192, remaining))
                if not packet:
                    video_socket_connected = False
                    break
                data += packet
            
            if not video_socket_connected:
                break

            frame_data = data[:msg_size]
            data = data[msg_size:]

            # Skip frames if queue is full (frame dropping for performance)
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()  # Remove old frame
                except queue.Empty:
                    pass

            # Decode and queue frame
            try:
                frame_buffer = pickle.loads(frame_data)
                frame = cv2.imdecode(frame_buffer, cv2.IMREAD_COLOR)
                if frame is not None:
                    frame_queue.put_nowait(frame)
                    frame_count += 1
            except queue.Full:
                pass  # Skip frame if queue full
            except Exception as e:
                print(f"Frame decode error: {e}")
                continue

        except Exception as e:
            print(f"Video processing error: {e}")
            video_socket_connected = False
            break

    video_socket_connected = False
    connection_status = "Video Disconnected"
    print("Video receiver ended")

def run_yolo_detection(frame):
    """Optimized YOLO detection with frame skipping"""
    global yolo_frame_counter
    
    if not yolo_enabled or model is None:
        return frame
    
    # Skip frames for performance
    yolo_frame_counter += 1
    if yolo_frame_counter % YOLO_SKIP_FRAMES != 0:
        return frame
    
    try:
        # Resize frame for faster YOLO processing
        h, w = frame.shape[:2]
        if w > 640:  # Resize large frames
            scale = 640 / w
            new_w, new_h = int(w * scale), int(h * scale)
            frame_small = cv2.resize(frame, (new_w, new_h))
        else:
            frame_small = frame
            scale = 1.0
        
        results = model(frame_small, conf=0.4, verbose=False)
        
        # Process results
        annotated = frame.copy()
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                target_classes = category_classes[choice]
                
                for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                    if int(cls) in target_classes:
                        # Scale coordinates back to original size
                        x1, y1, x2, y2 = (box / scale).int().tolist()
                        class_name = model.names[int(cls)]
                        label = f"{class_name} {conf:.2f}"
                        
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated, label, (x1, y1 - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add category info
        info_text = f"Detecting: {category_names[choice]}"
        cv2.putText(annotated, info_text, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated
        
    except Exception as e:
        print(f"YOLO detection error: {e}")
        return frame

def get_camera_surface():
    """Optimized camera frame processing with caching"""
    global processed_frame_cache
    
    try:
        frame = frame_queue.get_nowait()
        
        # Process with YOLO
        frame = run_yolo_detection(frame)
        
        # Resize and convert
        frame_resized = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_transposed = np.transpose(frame_rgb, (1, 0, 2))
        
        processed_frame_cache = pygame.surfarray.make_surface(frame_transposed)
        return processed_frame_cache
        
    except queue.Empty:
        # Return cached frame if no new frame available
        return processed_frame_cache
    except Exception as e:
        print(f"Camera surface error: {e}")
        return None

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

# Road boundaries (optimized with precomputed values)
ROADS = [
    (114, 215, 433, 245), (114, 300, 433, 330), (114, 388, 433, 418),
    (99, 230, 129, 403), (418, 230, 448, 403),
]

def is_on_road(x, y):
    """Optimized road checking"""
    for x1, y1, x2, y2 in ROADS:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False

def get_direction_from_movement(dx, dy):
    """Optimized direction calculation"""
    if abs(dx) > abs(dy):
        return 'right' if dx > 0 else 'left'
    else:
        return 'down' if dy > 0 else 'up'

# Initialize robot controller
robot_controller = RobotController()

# Start video receiver thread
video_thread = threading.Thread(target=video_receiver_thread, args=(robot_controller,), daemon=True)
video_thread.start()

# Connect control socket
robot_controller.connect_control()

# Initial state for visual car
x, y = 433, 403
speed = 1.2  # Slightly increased for smoother movement
direction = 'up'
keys_pressed = set()

# Pre-create font object
font = pygame.font.Font(None, 24)

# Main loop
running = True
frame_skip_counter = 0

print("🎮 Optimized Robot Control started!")
print(f"📡 Connecting to robot at {PI_IP}...")

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key in robot_controller.key_map:
                keys_pressed.add(event.key)
                robot_controller.send_command(robot_controller.key_map[event.key])
            elif event.key in STICKER_MAPPING:
                sticker_img, pos = STICKER_MAPPING[event.key]
                if sticker_img:  # Only add if image loaded successfully
                    placed_stickers[event.key] = (sticker_img, pos)
            elif event.key == pygame.K_c:
                placed_stickers.clear()
            elif event.key == pygame.K_q and yolo_enabled:
                choice = (choice % 3) + 1
                print(f"🎯 Changed to category: {category_names[choice]}")
            elif event.key == pygame.K_ESCAPE:
                running = False
        elif event.type == pygame.KEYUP:
            if event.key in robot_controller.key_map:
                keys_pressed.discard(event.key)
                if not any(key in robot_controller.key_map for key in keys_pressed):
                    robot_controller.send_stop()

    # Optimized visual car movement
    keys = pygame.key.get_pressed()
    dx = dy = 0
    if keys[pygame.K_UP] or keys[pygame.K_w]: dy -= speed
    if keys[pygame.K_DOWN] or keys[pygame.K_s]: dy += speed
    if keys[pygame.K_LEFT] or keys[pygame.K_a]: dx -= speed
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]: dx += speed
    
    if dx != 0 or dy != 0:
        new_x, new_y = x + dx, y + dy
        if is_on_road(new_x, new_y):
            x, y = new_x, new_y
            direction = get_direction_from_movement(dx, dy)

    # Drawing (optimized)
    if background:
        screen.blit(background, (0, 0))
    else:
        screen.fill((50, 50, 50))  # Fallback background

    # Draw stickers
    for sticker_img, position in placed_stickers.values():
        if sticker_img:
            rect = sticker_img.get_rect(center=position)
            screen.blit(sticker_img, rect)

    # Draw visual car
    if direction in rotated_stickers:
        sticker = rotated_stickers[direction]
        rect = sticker.get_rect(center=(x, y))
        screen.blit(sticker, rect)
    
    # Draw camera feed (optimized with frame skipping)
    camera_surface = get_camera_surface()
    if camera_surface:
        screen.blit(camera_surface, CAMERA_TOP_LEFT)
    else:
        # Simplified placeholder
        pygame.draw.rect(screen, (64, 64, 64), 
                        (CAMERA_TOP_LEFT[0], CAMERA_TOP_LEFT[1], CAMERA_WIDTH, CAMERA_HEIGHT))
        text = font.render("Camera Loading...", True, (255, 255, 255))
        text_rect = text.get_rect(center=(CAMERA_TOP_LEFT[0] + CAMERA_WIDTH//2, 
                                         CAMERA_TOP_LEFT[1] + CAMERA_HEIGHT//2))
        screen.blit(text, text_rect)
    
    # Draw camera border
    border_color = (0, 255, 0) if video_socket_connected else (255, 255, 255)
    pygame.draw.rect(screen, border_color, 
                    (CAMERA_TOP_LEFT[0], CAMERA_TOP_LEFT[1], CAMERA_WIDTH, CAMERA_HEIGHT), 2)
    
    # Draw status (cached font rendering)
    video_color = (0, 255, 0) if video_socket_connected else (255, 0, 0)
    control_color = (0, 255, 0) if control_socket_connected else (255, 0, 0)
    
    video_text = font.render(f"Video: {'Connected' if video_socket_connected else 'Disconnected'}", True, video_color)
    control_text = font.render(f"Control: {'Connected' if control_socket_connected else 'Disconnected'}", True, control_color)
    
    screen.blit(video_text, (10, HEIGHT - 90))
    screen.blit(control_text, (10, HEIGHT - 60))
    
    if yolo_enabled:
        yolo_text = font.render(f"YOLO: {category_names[choice]} (Press Q to change)", True, (0, 255, 0))
    else:
        yolo_text = font.render("YOLO: Disabled", True, (255, 0, 0))
    screen.blit(yolo_text, (10, HEIGHT - 30))

    pygame.display.flip()
    clock.tick(MAX_FPS)

# Cleanup
robot_controller.close()
pygame.quit()
print("Application closed cleanly")
sys.exit()