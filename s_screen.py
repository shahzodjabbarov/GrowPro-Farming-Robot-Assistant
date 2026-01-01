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
    choice = 3   # 1=Pumpkins, 2=Salads, 3=Strawberries it should be whatevr the user chooses in 3rd menu, fruit choosing button
    
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

# Pre-load and cache all images
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load and convert images once
print("Loading and optimizing images...")
background = pygame.image.load(os.path.join(script_dir, "map.jpg")).convert()
sticker_img = pygame.image.load(os.path.join(script_dir, "car.png")).convert_alpha()

# Pre-calculate car sticker rotations and cache them
sticker_scale = 7.5
sticker_size = (int(sticker_img.get_width() / sticker_scale), int(sticker_img.get_height() / sticker_scale))
sticker_original = pygame.transform.scale(sticker_img, sticker_size)

# Cache all rotated car stickers
car_stickers = {
    'up': sticker_original,
    'left': pygame.transform.rotate(sticker_original, 90),
    'down': pygame.transform.rotate(sticker_original, 180),
    'right': pygame.transform.rotate(sticker_original, -90)
}

# Load and scale vegetable stickers once
vegetable_images = {
    'strawberry': pygame.image.load(os.path.join(script_dir, "bad_strawberry.png")).convert_alpha(),
    'pumpkin': pygame.image.load(os.path.join(script_dir, "bad_pumpkin.png")).convert_alpha(),
    'lettuce': pygame.image.load(os.path.join(script_dir, "bad_lettuce.png")).convert_alpha()
}

# Pre-scale vegetable stickers
scale_factor = 0.1
vegetable_stickers = {}
for name, img in vegetable_images.items():
    new_size = (int(img.get_width() * scale_factor), int(img.get_height() * scale_factor))
    vegetable_stickers[name] = pygame.transform.scale(img, new_size)

# Dictionary to store placed stickers
placed_stickers = {}

# Optimized sticker mapping with pre-scaled images
STICKER_MAPPING = {
    pygame.K_1: (vegetable_stickers['strawberry'], (66, 326)),
    pygame.K_2: (vegetable_stickers['strawberry'], (65, 221)),
    pygame.K_3: (vegetable_stickers['pumpkin'], (340, 185)),
    pygame.K_4: (vegetable_stickers['pumpkin'], (400, 185)),
    pygame.K_5: (vegetable_stickers['lettuce'], (490, 424)),
    pygame.K_6: (vegetable_stickers['lettuce'], (445, 460)),
    pygame.K_7: (vegetable_stickers['lettuce'], (167, 355)),
    pygame.K_8: (vegetable_stickers['strawberry'], (192, 269))
}

# Pre-compiled road boundaries for faster collision detection
ROADS = [
    # Horizontal roads (x1, y1, x2, y2)
    (114, 215, 433, 245),
    (114, 300, 433, 330),
    (114, 388, 433, 418),
    # Vertical roads
    (99, 230, 129, 403),
    (418, 230, 448, 403),
]

def is_on_road(x, y):
    """Optimized road boundary check"""
    for x1, y1, x2, y2 in ROADS:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False

# Pre-calculate movement vectors for performance
MOVEMENT_VECTORS = {
    'up': (0, -1),
    'down': (0, 1),
    'left': (-1, 0),
    'right': (1, 0)
}

def get_next_position(x, y, direction, speed):
    """Optimized position calculation using pre-calculated vectors"""
    dx, dy = MOVEMENT_VECTORS[direction]
    return x + dx * speed, y + dy * speed

def get_direction_from_movement(dx, dy):
    """Optimized direction calculation"""
    if abs(dx) > abs(dy):
        return 'right' if dx > 0 else 'left'
    else:
        return 'down' if dy > 0 else 'up'

def run_yolo_detection(frame):
    """Optimized YOLO detection with reduced processing overhead"""
    if not yolo_enabled or model is None:
        return frame
    
    try:
        # Run detection with optimized parameters
        results = model(frame, conf=0.4, verbose=False)
        
        # Get target classes for current selection
        target_classes = category_classes[choice]
        original_result = results[0]
        
        # Early return if no detections
        if original_result.boxes is None or len(original_result.boxes) == 0:
            # Add category info overlay
            cv2.putText(frame, f"Detecting: {category_names[choice]}", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            return frame
        
        # Filter and draw detections in one pass
        boxes = original_result.boxes
        for i, cls in enumerate(boxes.cls):
            if int(cls) in target_classes:
                # Get box coordinates and confidence
                x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                conf = float(boxes.conf[i])
                class_name = model.names[int(cls)]
                label = f"{class_name} {conf:.2f}"
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label with background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Add category info overlay
        cv2.putText(frame, f"Detecting: {category_names[choice]}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
        
    except Exception as e:
        print(f"YOLO detection error: {e}")
        return frame

# Cache for camera surface to avoid repeated conversions
cached_camera_surface = None
last_frame_id = None

def capture_camera_frame():
    """Optimized camera frame processing with caching"""
    global current_frame, cached_camera_surface, last_frame_id
    
    with frame_lock:
        if current_frame is None:
            return None
        frame = current_frame.copy()
        frame_id = id(current_frame)  # Use object id for change detection
    
    # Skip processing if frame hasn't changed
    if frame_id == last_frame_id and cached_camera_surface is not None:
        return cached_camera_surface
    
    # Run YOLO detection
    frame = run_yolo_detection(frame)
    
    # Resize and convert color space
    frame_resized = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    
    # Convert to pygame surface using the most efficient method
    try:
        frame_transposed = np.transpose(frame_rgb, (1, 0, 2))
        camera_surface = pygame.surfarray.make_surface(frame_transposed)
        
        # Cache the result
        cached_camera_surface = camera_surface
        last_frame_id = frame_id
        
        return camera_surface
    except Exception as e:
        print(f"Camera surface conversion failed: {e}")
        return None

# Pre-create font objects
font_main = pygame.font.Font(None, 24)
font_status = pygame.font.Font(None, 20)

# Cache for status text surfaces to avoid re-rendering
status_text_cache = {}

def get_cached_text(text, font, color):
    """Get cached text surface or create new one"""
    cache_key = (text, font, color)
    if cache_key not in status_text_cache:
        status_text_cache[cache_key] = font.render(text, True, color)
    return status_text_cache[cache_key]

# Initial state
x, y = 433, 403
speed = 0.8
direction = 'up'

# Input handling optimization - track key states
key_states = {
    'up': False, 'down': False, 'left': False, 'right': False
}

# Frame counter for periodic operations
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
    
    # Draw background (only once per frame)
    screen.blit(background, (0, 0))

    # Event handling
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
                # Clear cached surfaces when category changes
                status_text_cache.clear()
            # Handle ESC key for exit
            elif event.key == pygame.K_ESCAPE:
                running = False

    # Optimized input handling
    keys = pygame.key.get_pressed()
    
    # Calculate movement delta
    dx = dy = 0
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

    # Draw placed vegetable stickers (batch operation)
    for sticker_img, position in placed_stickers.values():
        rect = sticker_img.get_rect(center=position)
        screen.blit(sticker_img, rect)

    # Draw car using cached rotated sticker
    sticker = car_stickers[direction]
    rect = sticker.get_rect(center=(x, y))
    screen.blit(sticker, rect)
    
    # Draw camera feed (optimized with caching)
    camera_frame = capture_camera_frame()
    if camera_frame:
        screen.blit(camera_frame, CAMERA_TOP_LEFT)
    else:
        # Draw placeholder rectangle
        pygame.draw.rect(screen, (64, 64, 64), 
                        (CAMERA_TOP_LEFT[0], CAMERA_TOP_LEFT[1], CAMERA_WIDTH, CAMERA_HEIGHT))
        
        # Show connection status (update less frequently)
        if frame_count % 30 == 0:  # Update every 30 frames (~0.5 seconds)
            status_lines = [connection_status, "Waiting for", "Raspberry Pi..."]
            for i, line in enumerate(status_lines):
                text = get_cached_text(line, font_main, (255, 255, 255))
                text_rect = text.get_rect(center=(CAMERA_TOP_LEFT[0] + CAMERA_WIDTH//2, 
                                                 CAMERA_TOP_LEFT[1] + CAMERA_HEIGHT//2 + (i-1)*30))
                screen.blit(text, text_rect)
    
    # Draw camera area border
    border_color = (0, 255, 0) if socket_connected else (255, 255, 255)
    pygame.draw.rect(screen, border_color, 
                    (CAMERA_TOP_LEFT[0], CAMERA_TOP_LEFT[1], CAMERA_WIDTH, CAMERA_HEIGHT), 2)
    
    # Draw status info (update less frequently to improve performance)
    if frame_count % 20 == 0:  # Update every 20 frames
        # Connection status
        conn_color = (0, 255, 0) if socket_connected else (255, 0, 0)
        conn_text = f"Pi Connection: {connection_status}"
        conn_surface = get_cached_text(conn_text, font_main, conn_color)
        
        # YOLO status
        if yolo_enabled:
            yolo_text = f"YOLO: {category_names[choice]} (Press Q to change)"
            yolo_color = (0, 255, 0)
        else:
            yolo_text = "YOLO: Disabled (Model not found)"
            yolo_color = (255, 0, 0)
        
        yolo_surface = get_cached_text(yolo_text, font_main, yolo_color)
    
    # Always blit the cached status surfaces
    if 'conn_surface' in locals():
        screen.blit(conn_surface, (10, HEIGHT - 60))
    if 'yolo_surface' in locals():
        screen.blit(yolo_surface, (10, HEIGHT - 30))

    pygame.display.flip()
    clock.tick(60)

# Cleanup
socket_connected = False  # Signal socket thread to stop
pygame.quit()
sys.exit()