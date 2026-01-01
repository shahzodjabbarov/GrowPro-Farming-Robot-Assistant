import pygame
import sys
import os
import math
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
pygame.display.set_caption("Free Street Navigation with Raspberry Pi YOLO Detection")
clock = pygame.time.Clock()

# Global variables for camera feed
current_frame = None
frame_lock = threading.Lock()
socket_connected = False
connection_status = "Disconnected"

# Camera display area coordinates
CAMERA_TOP_LEFT = (559, 161)
CAMERA_BOTTOM_RIGHT = (888, 479)
CAMERA_WIDTH = CAMERA_BOTTOM_RIGHT[0] - CAMERA_TOP_LEFT[0]
CAMERA_HEIGHT = CAMERA_BOTTOM_RIGHT[1] - CAMERA_TOP_LEFT[1]

# YOLO Setup
print("Loading YOLO model...")
try:
    # Clear memory
    gc.collect()
    
    # Load the trained model
    model = YOLO("prediction_ssppss.pt")
    
    # YOLO configuration
    choice = 1   # 1=Pumpkins, 2=Salads, 3=Strawberries it should be whatevr the user chooses in 3rd menu, fruit choosing button
    
    category_classes = {
        1: [0, 1],  # Pumpkins: Pumpkin A, Pumpkin Bro
        2: [2, 3],  # Salads: Salad A, Salad Bro
        3: [4, 5]   # Strawberries: Strawberries A, Strawberries Bro
    }
    
    category_names = {
        1: "Pumpkins",
        2: "Salads", 
        3: "Strawberries"
    }
    
    print(f"🎯 Selected category: {category_names[choice]}")
    print(f"   Detecting classes: {category_classes[choice]}")
    
    # Print model classes
    print("\n✅ Model classes loaded:")
    for cls_id, cls_name in model.names.items():
        print(f"Class {cls_id}: {cls_name}")
    
    yolo_enabled = True
    
except Exception as e:
    print(f"Warning: Could not load YOLO model: {e}")
    model = None
    yolo_enabled = False

def socket_receiver_thread():
    """Thread function to receive frames from Raspberry Pi"""
    global current_frame, socket_connected, connection_status
    
    try:
        # Create socket server
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(("0.0.0.0", 8485))
        server_socket.listen(1)
        
        connection_status = "Waiting for Raspberry Pi..."
        print("📡 Waiting for connection from Raspberry Pi on port 8485...")
        
        conn, addr = server_socket.accept()
        socket_connected = True
        connection_status = f"Connected to {addr[0]}"
        print(f"🔌 Connected to: {addr}")
        
        data = b""
        payload_size = struct.calcsize(">L")
        frame_counter = 0
        
        while socket_connected:
            try:
                # Read header (size)
                while len(data) < payload_size:
                    packet = conn.recv(4096)
                    if not packet:
                        break
                    data += packet
                if len(data) < payload_size:
                    continue

                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack(">L", packed_msg_size)[0]

                # Read frame data
                while len(data) < msg_size:
                    data += conn.recv(4096)
                frame_data = data[:msg_size]
                data = data[msg_size:]

                # Unpack and decode JPEG
                jpg_data = pickle.loads(frame_data)
                frame = cv2.imdecode(jpg_data, cv2.IMREAD_COLOR)
                
                if frame is None:
                    print("⚠️ Frame is None – decode error")
                    continue

                # Frame skip mechanism: only process every 3rd frame for YOLO
                frame_counter += 1
                
                # Always update the current frame for display
                with frame_lock:
                    current_frame = frame.copy()

            except Exception as e:
                print(f"❌ Error processing frame: {e}")
                time.sleep(0.1)  # Brief pause on error
                
    except Exception as e:
        print(f"❌ Socket error: {e}")
        connection_status = f"Error: {str(e)}"
    finally:
        socket_connected = False
        if 'conn' in locals():
            conn.close()
        if 'server_socket' in locals():
            server_socket.close()
        connection_status = "Disconnected"
        print("✅ Socket receiver ended")

# Start socket receiver thread
socket_thread = threading.Thread(target=socket_receiver_thread, daemon=True)
socket_thread.start()

# Load background and sticker
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

# Dictionary to store placed stickers
placed_stickers = {}

# Sticker mapping for number keys
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

# Define road boundaries
ROADS = [
    # Horizontal roads
    (114, 215, 433, 245, 'horizontal'),
    (114, 300, 433, 330, 'horizontal'),
    (114, 388, 433, 418, 'horizontal'),
    # Vertical roads
    (99, 230, 129, 403, 'vertical'),
    (418, 230, 448, 403, 'vertical'),
]

def is_on_road(x, y):
    """Check if position (x, y) is within any road boundary"""
    for road in ROADS:
        x1, y1, x2, y2, road_type = road
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False

def get_next_position(x, y, direction, speed):
    """Calculate next position based on current direction"""
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
    """Determine direction based on movement"""
    dx = new_x - old_x
    dy = new_y - old_y
    
    if abs(dx) > abs(dy):
        return 'right' if dx > 0 else 'left'
    else:
        return 'down' if dy > 0 else 'up'

def get_rotated_sticker(direction):
    """Rotate sticker based on direction"""
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
    """Run YOLO detection on frame and return annotated frame"""
    if not yolo_enabled or model is None:
        return frame
    
    try:
        # Run detection with confidence threshold
        results = model(frame, conf=0.65)
        
        # Filter results to only show selected category
        filtered_boxes = []
        original_result = results[0]
        
        if original_result.boxes is not None:
            target_classes = category_classes[choice]
            
            # Filter boxes based on selected category
            for i, cls in enumerate(original_result.boxes.cls):
                if int(cls) in target_classes:
                    filtered_boxes.append(i)
        
        # Create annotated frame
        annotated = frame.copy()
        
        # Draw filtered detections
        if filtered_boxes and original_result.boxes is not None:
            filtered_boxes_tensor = original_result.boxes.xyxy[filtered_boxes]
            filtered_conf = original_result.boxes.conf[filtered_boxes]
            filtered_cls = original_result.boxes.cls[filtered_boxes]
            
            # Draw bounding boxes and labels
            for box, conf, cls in zip(filtered_boxes_tensor, filtered_conf, filtered_cls):
                x1, y1, x2, y2 = map(int, box)
                class_name = model.names[int(cls)]
                label = f"{class_name} {conf:.2f}"
                
                # Draw bounding box (green)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), (0, 255, 0), -1)
                
                # Draw label text
                cv2.putText(annotated, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Add category info to display
        if yolo_enabled:
            info_text = f"Detecting: {category_names[1]}"
            cv2.putText(annotated, info_text, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated
        
    except Exception as e:
        print(f"YOLO detection error: {e}")
        return frame
def capture_camera_frame():
    """Process camera frame from Raspberry Pi for pygame display with YOLO detection"""
    global current_frame
    
    with frame_lock:
        if current_frame is None:
            return None
        frame = current_frame.copy()
    
    # Run YOLO detection on the frame
    frame = run_yolo_detection(frame)
    
    # Resize to fit camera display area
    frame_resized = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
    
    # Convert BGR to RGB (OpenCV uses BGR, pygame expects RGB)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    
    # Convert numpy array to pygame surface
    # Method 1: Direct conversion
    try:
        # Transpose the array for pygame (height, width, channels) -> (width, height, channels)
        frame_transposed = np.transpose(frame_rgb, (1, 0, 2))
        camera_surface = pygame.surfarray.make_surface(frame_transposed)
        return camera_surface
    except Exception as e:
        print(f"Method 1 failed: {e}")
        
        # Method 2: Using pygame.image.fromstring
        try:
            h, w, c = frame_rgb.shape
            raw_data = frame_rgb.tobytes()
            camera_surface = pygame.image.fromstring(raw_data, (w, h), 'RGB')
            return camera_surface
        except Exception as e2:
            print(f"Method 2 failed: {e2}")
            
            # Method 3: Manual pixel-by-pixel (slower but reliable)
            try:
                h, w = frame_rgb.shape[:2]
                camera_surface = pygame.Surface((w, h))
                pygame.surfarray.blit_array(camera_surface, np.transpose(frame_rgb, (1, 0, 2)))
                return camera_surface
            except Exception as e3:
                print(f"All methods failed: {e3}")
                return None
            

# Initial state
x, y = 433, 403
speed = 0.5
direction = 'up'
# Main loop
running = True
print("🎮 Game started! Controls:")
print("   Arrow keys or WASD: Move car")
print("   Number keys 1-8: Place vegetable stickers")
print("   D key: Clear all stickers")
print("   Q key: Change YOLO detection category")
print("   ESC: Exit")
print("\n📡 Waiting for Raspberry Pi connection on port 8485...")

while running:
    screen.blit(background, (0, 0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            # Handle number key presses for placing stickers
            if event.key in STICKER_MAPPING:
                sticker_img, position = STICKER_MAPPING[event.key]
                placed_stickers[event.key] = (sticker_img, position)
            # Handle 'd' key for clearing all stickers
            elif event.key == pygame.K_d:
                placed_stickers.clear()
            # Handle 'q' key for changing YOLO category
            elif event.key == pygame.K_q and yolo_enabled:
                choice = (choice % 3) + 1  # Cycle through 1, 2, 3
                print(f"🎯 Changed to category: {category_names[choice]}")
            # Handle ESC key for exit
            elif event.key == pygame.K_ESCAPE:
                running = False

    # Handle input for free movement
    keys = pygame.key.get_pressed()
    old_x, old_y = x, y
    new_x, new_y = x, y
    
    # Calculate intended movement
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        new_x, new_y = get_next_position(x, y, 'up', speed)
    elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
        new_x, new_y = get_next_position(x, y, 'down', speed)
    elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
        new_x, new_y = get_next_position(x, y, 'left', speed)
    elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        new_x, new_y = get_next_position(x, y, 'right', speed)
    
    # Only move if the new position is on a road
    if new_x != x or new_y != y:
        if is_on_road(new_x, new_y):
            x, y = new_x, new_y
            direction = get_direction_from_movement(old_x, old_y, x, y)
    
    # Alternative: Smooth movement with multiple direction keys
    dx, dy = 0, 0
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        dy -= speed
    if keys[pygame.K_DOWN] or keys[pygame.K_s]:
        dy += speed
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        dx -= speed
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        dx += speed
    
    # Apply smooth movement if within roads
    if dx != 0 or dy != 0:
        new_x, new_y = x + dx, y + dy
        if is_on_road(new_x, new_y):
            old_x, old_y = x, y
            x, y = new_x, new_y
            direction = get_direction_from_movement(old_x, old_y, x, y)

    # Draw placed vegetable stickers first
    for key, (sticker_img, position) in placed_stickers.items():
        rect = sticker_img.get_rect(center=position)
        screen.blit(sticker_img, rect)

    # Draw car
    sticker = get_rotated_sticker(direction)
    rect = sticker.get_rect(center=(x, y))
    screen.blit(sticker, rect)
    
    # Draw camera feed with YOLO detection
    camera_frame = capture_camera_frame()
    if camera_frame:
        screen.blit(camera_frame, CAMERA_TOP_LEFT)
    else:
        # Draw placeholder rectangle if no camera feed
        pygame.draw.rect(screen, (64, 64, 64), 
                        (CAMERA_TOP_LEFT[0], CAMERA_TOP_LEFT[1], CAMERA_WIDTH, CAMERA_HEIGHT))
        font = pygame.font.Font(None, 24)
        
        # Show connection status
        status_lines = [
            connection_status,
            "Waiting for",
            "Raspberry Pi..."
        ]
        
        for i, line in enumerate(status_lines):
            text = font.render(line, True, (255, 255, 255))
            text_rect = text.get_rect(center=(CAMERA_TOP_LEFT[0] + CAMERA_WIDTH//2, 
                                             CAMERA_TOP_LEFT[1] + CAMERA_HEIGHT//2 + (i-1)*30))
            screen.blit(text, text_rect)
    
    # Draw camera area border
    border_color = (0, 255, 0) if socket_connected else (255, 255, 255)
    pygame.draw.rect(screen, border_color, 
                    (CAMERA_TOP_LEFT[0], CAMERA_TOP_LEFT[1], CAMERA_WIDTH, CAMERA_HEIGHT), 2)
    
    # Draw status info on screen
    font = pygame.font.Font(None, 24)
    
    # Connection status
    conn_color = (0, 255, 0) if socket_connected else (255, 0, 0)
    conn_text = f"Pi Connection: {connection_status}"
    conn_surface = font.render(conn_text, True, conn_color)
    screen.blit(conn_surface, (10, HEIGHT - 60))
    
    # YOLO status
    if yolo_enabled:
        yolo_text = f"YOLO: {category_names[choice]} (Press Q to change)"
        yolo_color = (0, 255, 0)
    else:
        yolo_text = "YOLO: Disabled (Model not found)"
        yolo_color = (255, 0, 0)
    
    yolo_surface = font.render(yolo_text, True, yolo_color)
    screen.blit(yolo_surface, (10, HEIGHT - 30))

    pygame.display.flip()
    clock.tick(60)

# Cleanup
socket_connected = False  # Signal socket thread to stop
pygame.quit()
sys.exit()