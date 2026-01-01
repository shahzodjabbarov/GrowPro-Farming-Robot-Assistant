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
from queue import Queue
import weakref

# Setup
pygame.init()
WIDTH, HEIGHT = 955, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Free Street Navigation with Raspberry Pi YOLO Detection")
clock = pygame.time.Clock()

# Global variables for camera feed - using thread-safe approach
frame_lock = threading.Lock()
current_frame = None
current_camera_surface = None
socket_connected = False
connection_status = "Disconnected"
last_yolo_time = 0
YOLO_INTERVAL = 0.1  # Run YOLO every 100ms instead of every frame

# Camera display area coordinates
CAMERA_TOP_LEFT = (559, 161)
CAMERA_BOTTOM_RIGHT = (888, 479)
CAMERA_WIDTH = CAMERA_BOTTOM_RIGHT[0] - CAMERA_TOP_LEFT[0]
CAMERA_HEIGHT = CAMERA_BOTTOM_RIGHT[1] - CAMERA_TOP_LEFT[1]

# Pre-calculated camera surface
camera_surface = pygame.Surface((CAMERA_WIDTH, CAMERA_HEIGHT))
placeholder_surface = None

# YOLO Setup
print("Loading YOLO model...")
try:
    # Clear memory first
    gc.collect()
    
    # Load the trained model
    model = YOLO("prediction_ssppss.pt")
    model.fuse()  # Fuse model for faster inference
    
    # YOLO configuration
    choice = 2  # 1=Pumpkins, 2=Salads, 3=Strawberries
    
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

def create_placeholder_surface():
    """Create placeholder surface once"""
    global placeholder_surface
    if placeholder_surface is None:
        placeholder_surface = pygame.Surface((CAMERA_WIDTH, CAMERA_HEIGHT))
        placeholder_surface.fill((64, 64, 64))
        
        font = pygame.font.Font(None, 24)
        status_lines = ["Waiting for", "Raspberry Pi..."]
        
        for i, line in enumerate(status_lines):
            text = font.render(line, True, (255, 255, 255))
            text_rect = text.get_rect(center=(CAMERA_WIDTH//2, CAMERA_HEIGHT//2 + (i-0.5)*30))
            placeholder_surface.blit(text, text_rect)

def socket_receiver_thread():
    """Improved thread function to receive frames from Raspberry Pi"""
    global socket_connected, connection_status, current_frame
    
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
        
        # Set socket timeout to prevent blocking
        conn.settimeout(1.0)
        
        data = b""
        payload_size = struct.calcsize(">L")
        
        while socket_connected:
            try:
                # Read header (size)
                while len(data) < payload_size:
                    try:
                        packet = conn.recv(4096)
                        if not packet:
                            break
                        data += packet
                    except socket.timeout:
                        continue
                    except Exception as e:
                        print(f"Error receiving header: {e}")
                        break
                        
                if len(data) < payload_size:
                    continue

                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack(">L", packed_msg_size)[0]

                # Read frame data
                while len(data) < msg_size:
                    try:
                        remaining = msg_size - len(data)
                        chunk_size = min(4096, remaining)
                        packet = conn.recv(chunk_size)
                        if not packet:
                            break
                        data += packet
                    except socket.timeout:
                        continue
                    except Exception as e:
                        print(f"Error receiving frame data: {e}")
                        break

                if len(data) < msg_size:
                    continue

                frame_data = data[:msg_size]
                data = data[msg_size:]

                # Unpack and decode JPEG
                try:
                    jpg_data = pickle.loads(frame_data)
                    frame = cv2.imdecode(jpg_data, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        # Thread-safe frame update
                        with frame_lock:
                            current_frame = frame.copy()
                            
                except Exception as e:
                    print(f"Error decoding frame: {e}")
                    continue

            except Exception as e:
                print(f"❌ Error processing frame: {e}")
                time.sleep(0.1)
                
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

def run_yolo_detection(frame):
    """Optimized YOLO detection with timing control"""
    global last_yolo_time
    
    if not yolo_enabled or model is None:
        return frame
    
    current_time = time.time()
    if current_time - last_yolo_time < YOLO_INTERVAL:
        return frame  # Skip YOLO processing
    
    try:
        last_yolo_time = current_time
        
        # Resize frame for faster processing
        h, w = frame.shape[:2]
        if w > 640:  # Reduce resolution for faster processing
            scale = 640 / w
            new_w, new_h = int(w * scale), int(h * scale)
            frame_small = cv2.resize(frame, (new_w, new_h))
        else:
            frame_small = frame
            scale = 1.0
        
        # Run detection with lower confidence for faster processing
        results = model(frame_small, conf=0.3, verbose=False)
        
        # Filter results to only show selected category
        original_result = results[0]
        
        if original_result.boxes is not None:
            target_classes = category_classes[choice]
            
            # Create annotated frame
            annotated = frame.copy()
            
            # Draw filtered detections
            for i, cls in enumerate(original_result.boxes.cls):
                if int(cls) in target_classes:
                    box = original_result.boxes.xyxy[i]
                    conf = original_result.boxes.conf[i]
                    
                    # Scale coordinates back if we resized
                    x1, y1, x2, y2 = map(int, box / scale if scale != 1.0 else box)
                    class_name = model.names[int(cls)]
                    label = f"{class_name} {conf:.2f}"
                    
                    # Draw bounding box (green)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label background
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(annotated, (x1, y1 - label_size[1] - 5), 
                                 (x1 + label_size[0], y1), (0, 255, 0), -1)
                    
                    # Draw label text
                    cv2.putText(annotated, label, (x1, y1 - 2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            return annotated
        
        return frame
        
    except Exception as e:
        print(f"YOLO detection error: {e}")
        return frame

def update_camera_surface():
    """Update camera surface with latest frame"""
    global current_camera_surface
    
    with frame_lock:
        if current_frame is not None:
            try:
                # Create a copy of the current frame
                frame = current_frame.copy()
                
                # Run YOLO detection (with timing control)
                frame = run_yolo_detection(frame)
                
                # Resize to fit camera display area
                frame_resized = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                
                # Convert to pygame surface efficiently
                frame_transposed = np.transpose(frame_rgb, (1, 0, 2))
                
                # Create new surface for this frame
                temp_surface = pygame.Surface((CAMERA_WIDTH, CAMERA_HEIGHT))
                pygame.surfarray.blit_array(temp_surface, frame_transposed)
                
                # Update the current camera surface
                current_camera_surface = temp_surface
                
                return True
                
            except Exception as e:
                print(f"Frame processing error: {e}")
                return False
    
    return False

# Start socket receiver thread
socket_thread = threading.Thread(target=socket_receiver_thread, daemon=True)
socket_thread.start()

# Load and cache resources
script_dir = os.path.dirname(os.path.abspath(__file__))
background = pygame.image.load(os.path.join(script_dir, "map.jpg")).convert()
sticker_img = pygame.image.load(os.path.join(script_dir, "car.png")).convert_alpha()
sticker_original = pygame.transform.scale(sticker_img, 
    (int(sticker_img.get_width() // 7.5), int(sticker_img.get_height() // 7.5)))

# Pre-calculate rotated stickers
rotated_stickers = {
    'up': sticker_original,
    'left': pygame.transform.rotate(sticker_original, 90),
    'down': pygame.transform.rotate(sticker_original, 180),
    'right': pygame.transform.rotate(sticker_original, -90)
}

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

# Define road boundaries (pre-calculated)
ROADS = [
    (114, 215, 433, 245, 'horizontal'),
    (114, 300, 433, 330, 'horizontal'),
    (114, 388, 433, 418, 'horizontal'),
    (99, 230, 129, 403, 'vertical'),
    (418, 230, 448, 403, 'vertical'),
]

def is_on_road(x, y):
    """Optimized road boundary check"""
    for x1, y1, x2, y2, _ in ROADS:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False

def get_direction_from_movement(dx, dy):
    """Optimized direction calculation"""
    if abs(dx) > abs(dy):
        return 'right' if dx > 0 else 'left'
    else:
        return 'down' if dy > 0 else 'up'

# Create placeholder surface
create_placeholder_surface()

# Pre-create font objects
font = pygame.font.Font(None, 24)

# Initial state
x, y = 433, 403
speed = 1.2  # Slightly increased for smoother movement
direction = 'up'
frame_count = 0

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
    frame_count += 1
    
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key in STICKER_MAPPING:
                sticker_img, position = STICKER_MAPPING[event.key]
                placed_stickers[event.key] = (sticker_img, position)
            elif event.key == pygame.K_d:
                placed_stickers.clear()
            elif event.key == pygame.K_q and yolo_enabled:
                choice = (choice % 3) + 1
                print(f"🎯 Changed to category: {category_names[choice]}")
            elif event.key == pygame.K_ESCAPE:
                running = False

    # Handle smooth movement
    keys = pygame.key.get_pressed()
    dx, dy = 0, 0
    
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        dy -= speed
    if keys[pygame.K_DOWN] or keys[pygame.K_s]:
        dy += speed
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        dx -= speed
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        dx += speed
    
    # Apply movement if within roads
    if dx != 0 or dy != 0:
        new_x, new_y = x + dx, y + dy
        if is_on_road(new_x, new_y):
            x, y = new_x, new_y
            direction = get_direction_from_movement(dx, dy)

    # Clear screen and draw background
    screen.blit(background, (0, 0))

    # Draw placed vegetable stickers
    for sticker_img, position in placed_stickers.values():
        sticker_rect = sticker_img.get_rect(center=position)
        screen.blit(sticker_img, sticker_rect)

    # Draw car with pre-calculated rotation
    sticker = rotated_stickers[direction]
    car_rect = sticker.get_rect(center=(x, y))
    screen.blit(sticker, car_rect)
    
    # Update and display camera feed
    if socket_connected:
        # Try to update camera surface (non-blocking)
        update_camera_surface()
        
        # Display the current camera surface if available
        if current_camera_surface is not None:
            screen.blit(current_camera_surface, CAMERA_TOP_LEFT)
        else:
            screen.blit(placeholder_surface, CAMERA_TOP_LEFT)
    else:
        screen.blit(placeholder_surface, CAMERA_TOP_LEFT)
    
    # Draw camera border
    border_color = (0, 255, 0) if socket_connected else (255, 255, 255)
    pygame.draw.rect(screen, border_color, 
                    (CAMERA_TOP_LEFT[0], CAMERA_TOP_LEFT[1], CAMERA_WIDTH, CAMERA_HEIGHT), 2)
    
    # Draw status info (only every 10 frames to reduce text rendering overhead)
    if frame_count % 10 == 0:
        # Connection status
        conn_color = (0, 255, 0) if socket_connected else (255, 0, 0)
        conn_text = f"Pi Connection: {connection_status}"
        conn_surface = font.render(conn_text, True, conn_color)
        
        # YOLO status
        if yolo_enabled:
            yolo_text = f"YOLO: {category_names[choice]} (Press Q to change)"
            yolo_color = (0, 255, 0)
        else:
            yolo_text = "YOLO: Disabled (Model not found)"
            yolo_color = (255, 0, 0)
        
        yolo_surface = font.render(yolo_text, True, yolo_color)
        
        # Cache the rendered text surfaces
        status_surfaces = [
            (conn_surface, (10, HEIGHT - 60)),
            (yolo_surface, (10, HEIGHT - 30))
        ]
    
    # Draw cached status surfaces
    if 'status_surfaces' in locals():
        for surface, pos in status_surfaces:
            screen.blit(surface, pos)

    pygame.display.flip()
    clock.tick(60)  # Maintain 60 FPS

# Cleanup
socket_connected = False
pygame.quit()
sys.exit()