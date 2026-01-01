import cv2
import numpy as np
import time
import os
import sys

class LaneDetectionTester:
    def __init__(self):
        self.navigation_state = {
            'last_direction': 'straight',
            'direction_confidence': 0,
            'stable_navigation': True,
            'last_decision_time': 0,
            'decision_interval': 0.5,
            'left_line_history': [],
            'right_line_history': [],
            'line_history_size': 10  # Keep history for stable line detection
        }
        
        # Motor speed constants
        self.MIN_SPEED = 15
        self.MAX_SPEED = 40
        self.FULL_TURN_SPEED = 35
        
        # Navigation output
        self.current_command = {
            'direction': 'straight',
            'left_speed': 0,
            'right_speed': 0,
            'confidence': 0,
            'left_angle': 0,
            'right_angle': 0,
            'angle_difference': 0
        }
        
        # Stable line parameters
        self.current_left_line = None
        self.current_right_line = None

    def create_triangular_mask(self, image):
        """
        Create triangular mask for lane area segmentation as mentioned in the research
        """
        h, w = image.shape
        
        # Create triangular ROI mask
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Define triangle points - from bottom corners to middle-top
        bottom_left = [int(w * 0.1), h]      # Start from lower left corner
        bottom_right = [int(w * 0.9), h]     # Start from lower right corner
        top_center = [int(w * 0.5), int(h * 0.6)]  # End at middle-top
        
        roi_points = np.array([
            bottom_left,
            bottom_right, 
            top_center
        ], dtype=np.int32)
        
        cv2.fillPoly(roi_mask, [roi_points], 255)
        
        return roi_mask, roi_points

    def apply_canny_edge_detection(self, image):
        """
        Apply Canny Edge Detector to detect sharp luminosity changes (white/yellow lanes vs gray/black road)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection with adaptive thresholds
        # Lower threshold for detecting weak edges (subtle lane markings)
        # Higher threshold for strong edges
        edges = cv2.Canny(blur, 50, 150, apertureSize=3)
        
        return edges

    def detect_lane_lines_hough(self, edges, original_frame):
        """
        Use Hough Transform to extract lanes as straight lines from bottom corners
        """
        h, w = edges.shape
        
        # Hough line detection with parameters optimized for lane detection
        lines = cv2.HoughLinesP(
            edges,
            rho=2,              # Distance resolution in pixels
            theta=np.pi/180,    # Angle resolution in radians
            threshold=100,      # Minimum votes threshold
            minLineLength=50,   # Minimum line length
            maxLineGap=50       # Maximum gap between line segments
        )
        
        # Separate lines into left and right based on position and angle
        left_lines = []
        right_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate line angle
                if x2 - x1 == 0:  # Vertical line
                    continue
                    
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                # Filter lines that start from bottom area and go up
                line_center_x = (x1 + x2) / 2
                line_bottom_y = max(y1, y2)
                
                # Only consider lines that start from bottom 40% of image
                if line_bottom_y < h * 0.6:
                    continue
                
                # Classify left vs right lines based on position and angle
                if line_center_x < w * 0.5:  # Left side
                    # Left lines should have positive slope (going up-right)
                    if -80 < angle < -20:  # Steep upward slope for left lane
                        left_lines.append((line[0], angle))
                else:  # Right side
                    # Right lines should have negative slope (going up-left)  
                    if 20 < angle < 80:   # Steep upward slope for right lane
                        right_lines.append((line[0], angle))
        
        return left_lines, right_lines

    def get_stable_line(self, lines, side, frame_height, frame_width):
        """
        Get stable line by averaging recent detections and filtering by consistency
        """
        if not lines:
            return None, 0
            
        # Find the best line based on position consistency and angle
        best_line = None
        best_angle = 0
        best_score = 0
        
        for line_data, angle in lines:
            x1, y1, x2, y2 = line_data
            
            # Ensure line goes from bottom corner area towards middle
            if side == 'left':
                # Left line should start from left bottom area
                if min(x1, x2) < frame_width * 0.3 and max(y1, y2) > frame_height * 0.7:
                    # Score based on how well it fits expected left lane pattern
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    angle_score = max(0, 60 - abs(angle + 45))  # Prefer ~45 degree slope
                    score = length * angle_score
                    
                    if score > best_score:
                        best_score = score
                        best_line = line_data
                        best_angle = angle
                        
            else:  # right side
                # Right line should start from right bottom area  
                if max(x1, x2) > frame_width * 0.7 and max(y1, y2) > frame_height * 0.7:
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    angle_score = max(0, 60 - abs(angle - 45))  # Prefer ~45 degree slope
                    score = length * angle_score
                    
                    if score > best_score:
                        best_score = score
                        best_line = line_data
                        best_angle = angle
        
        return best_line, best_angle

    def update_stable_lines(self, left_lines, right_lines, frame_height, frame_width):
        """
        Update stable lines using history to prevent blinking
        """
        # Get best current lines
        new_left_line, left_angle = self.get_stable_line(left_lines, 'left', frame_height, frame_width)
        new_right_line, right_angle = self.get_stable_line(right_lines, 'right', frame_height, frame_width)
        
        # Update line history
        if new_left_line is not None:
            self.navigation_state['left_line_history'].append((new_left_line, left_angle))
            if len(self.navigation_state['left_line_history']) > self.navigation_state['line_history_size']:
                self.navigation_state['left_line_history'].pop(0)
        
        if new_right_line is not None:
            self.navigation_state['right_line_history'].append((new_right_line, right_angle))
            if len(self.navigation_state['right_line_history']) > self.navigation_state['line_history_size']:
                self.navigation_state['right_line_history'].pop(0)
        
        # Use averaged stable lines if we have enough history
        if len(self.navigation_state['left_line_history']) >= 3:
            # Average recent detections for stability
            recent_left = self.navigation_state['left_line_history'][-3:]
            avg_left_line = np.mean([line for line, angle in recent_left], axis=0).astype(int)
            avg_left_angle = np.mean([angle for line, angle in recent_left])
            self.current_left_line = avg_left_line
            self.current_command['left_angle'] = avg_left_angle
        else:
            if new_left_line is not None:
                self.current_left_line = new_left_line
                self.current_command['left_angle'] = left_angle
        
        if len(self.navigation_state['right_line_history']) >= 3:
            recent_right = self.navigation_state['right_line_history'][-3:]
            avg_right_line = np.mean([line for line, angle in recent_right], axis=0).astype(int)
            avg_right_angle = np.mean([angle for line, angle in recent_right])
            self.current_right_line = avg_right_line
            self.current_command['right_angle'] = avg_right_angle
        else:
            if new_right_line is not None:
                self.current_right_line = new_right_line
                self.current_command['right_angle'] = right_angle

    def calculate_navigation_from_angles(self, frame_width):
        """
        Calculate navigation based on line angles and their steepness difference
        """
        direction = "straight"
        confidence = 0
        
        left_angle = self.current_command['left_angle']
        right_angle = self.current_command['right_angle']
        
        has_left = self.current_left_line is not None
        has_right = self.current_right_line is not None
        
        if has_left and has_right:
            # Both lanes detected - analyze angle difference
            angle_diff = right_angle - left_angle  # Should be positive for normal lanes
            self.current_command['angle_difference'] = angle_diff
            
            # Normal lane configuration: left ~-45°, right ~+45°, diff ~90°
            expected_diff = 90
            diff_error = angle_diff - expected_diff
            
            # Determine steering based on angle difference
            if abs(diff_error) < 15:  # Both lanes roughly parallel
                direction = "straight"
                confidence = 8
            elif diff_error > 20:  # Right lane steeper than expected
                direction = "slight_left"
                confidence = 7
            elif diff_error < -20:  # Left lane steeper than expected  
                direction = "slight_right"
                confidence = 7
            elif diff_error > 0:
                direction = "slight_left"
                confidence = 5
            else:
                direction = "slight_right"
                confidence = 5
                
        elif has_left:
            # Only left lane - steer based on left line steepness
            if left_angle < -60:  # Very steep left line
                direction = "slight_right"  # Move away from steep left edge
                confidence = 6
            elif left_angle < -30:  # Moderate left line
                direction = "slight_right"
                confidence = 4
            else:
                direction = "straight"
                confidence = 3
                
        elif has_right:
            # Only right lane - steer based on right line steepness
            if right_angle > 60:  # Very steep right line
                direction = "slight_left"   # Move away from steep right edge
                confidence = 6
            elif right_angle > 30:  # Moderate right line
                direction = "slight_left"
                confidence = 4
            else:
                direction = "straight"
                confidence = 3
        else:
            # No lanes detected
            direction = self.navigation_state['last_direction']
            confidence = 1
        
        return direction, confidence

    def calculate_motor_speeds(self, direction, confidence):
        """
        Calculate motor speeds based on direction and confidence
        """
        base_speed = self.MIN_SPEED + (confidence / 10.0) * (self.MAX_SPEED - self.MIN_SPEED)
        base_speed = max(self.MIN_SPEED, min(self.MAX_SPEED, base_speed))
        
        if direction == "straight":
            return int(base_speed), int(base_speed)
        elif direction == "slight_left":
            # Reduce left motor proportional to angle difference
            angle_diff = abs(self.current_command['angle_difference'])
            reduction = min(angle_diff / 90.0, 0.4) * (base_speed - self.MIN_SPEED)
            left_speed = max(self.MIN_SPEED, base_speed - reduction)
            return int(left_speed), int(base_speed)
        elif direction == "slight_right":
            angle_diff = abs(self.current_command['angle_difference'])
            reduction = min(angle_diff / 90.0, 0.4) * (base_speed - self.MIN_SPEED)
            right_speed = max(self.MIN_SPEED, base_speed - reduction)
            return int(base_speed), int(right_speed)
        else:
            return 0, 0

    def should_make_decision(self):
        """
        Check if enough time has passed to make a new navigation decision
        """
        current_time = time.time()
        if current_time - self.navigation_state['last_decision_time'] >= self.navigation_state['decision_interval']:
            self.navigation_state['last_decision_time'] = current_time
            return True
        return False

    def process_frame(self, frame):
        """
        Process frame using the 5-step lane detection approach
        """
        # Step 1: Apply Canny Edge Detection
        edges = self.apply_canny_edge_detection(frame)
        
        # Step 2: Apply triangular mask for lane area segmentation
        roi_mask, roi_points = self.create_triangular_mask(edges)
        masked_edges = cv2.bitwise_and(edges, roi_mask)
        
        # Step 3: Use Hough Transform to detect lane lines
        left_lines, right_lines = self.detect_lane_lines_hough(masked_edges, frame)
        
        # Step 4: Update stable lines (prevent blinking)
        h, w = frame.shape[:2]
        self.update_stable_lines(left_lines, right_lines, h, w)
        
        # Step 5: Make navigation decisions every 0.5 seconds
        if self.should_make_decision():
            direction, confidence = self.calculate_navigation_from_angles(w)
            left_speed, right_speed = self.calculate_motor_speeds(direction, confidence)
            
            self.current_command.update({
                'direction': direction,
                'left_speed': left_speed,
                'right_speed': right_speed,
                'confidence': confidence
            })
            
            self.navigation_state['last_direction'] = direction
        
        # Step 6: Visualization with stable lines
        self.add_visualization(frame, roi_points)
        
        return frame

    def add_visualization(self, frame, roi_points):
        """
        Add visualization overlay with stable lane lines
        """
        # Draw triangular ROI
        cv2.polylines(frame, [roi_points], True, (255, 255, 0), 2)
        
        # Draw stable lane lines (light green as mentioned in research)
        if self.current_left_line is not None:
            x1, y1, x2, y2 = self.current_left_line
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 100), 4)  # Light green
            # Add angle text
            cv2.putText(frame, f"L: {self.current_command['left_angle']:.1f}°", 
                       (x1-20, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 2)
        
        if self.current_right_line is not None:
            x1, y1, x2, y2 = self.current_right_line
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 100), 4)  # Light green
            cv2.putText(frame, f"R: {self.current_command['right_angle']:.1f}°", 
                       (x2+10, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 2)
        
        # Draw center line
        h, w = frame.shape[:2]
        cv2.line(frame, (w//2, 0), (w//2, h), (255, 0, 255), 2)
        
        # Add navigation info
        cmd = self.current_command
        y_offset = 30
        
        cv2.putText(frame, f"Direction: {cmd['direction']}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        
        cv2.putText(frame, f"Left Motor: {cmd['left_speed']}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30
        
        cv2.putText(frame, f"Right Motor: {cmd['right_speed']}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += 30
        
        cv2.putText(frame, f"Confidence: {cmd['confidence']}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        
        cv2.putText(frame, f"Angle Diff: {cmd.get('angle_difference', 0):.1f}°", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def get_navigation_command(self):
        """
        Get current navigation command
        """
        return self.current_command

def get_video_file():
    """
    Get video file path from user input or command line argument
    """
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = input("Enter video file path (or press Enter for default 'test_video.mp4'): ")
        if not video_path.strip():
            video_path = "vid4.mp4"
    
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found!")
        print("Please make sure the video file exists in the specified path.")
        return None
    
    return video_path

def main():
    # Get video file path
    video_path = "C://Users//uaser//Desktop//vid4.mp4"
    if video_path is None:
        return
    
    lane_detector = LaneDetectionTester()
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        print("Supported formats: .mp4, .avi, .mov, .mkv, .wmv, .flv")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count_total / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("Angle-Based Lane Detection System - Video Mode")
    print("=" * 60)
    print(f"Video: {os.path.basename(video_path)}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Duration: {duration:.1f} seconds ({frame_count_total} frames)")
    print("=" * 60)
    print("Features:")
    print("- Triangular ROI mask for lane area segmentation")
    print("- Canny edge detection for luminosity changes")
    print("- Hough transform for line extraction")
    print("- Angle-based navigation (steepness analysis)")
    print("- Stable line detection (no blinking)")
    print("- Decision interval: 0.5 seconds")
    print("=" * 60)
    print("Controls:")
    print("- Press 'q' to quit")
    print("- Press 's' to save current frame")
    print("- Press 'p' to pause/resume")
    print("- Press 'r' to restart video")
    print("=" * 60)
    
    frame_count = 0
    last_print_time = time.time()
    paused = False
    
    # Calculate frame delay for real-time playback
    target_fps = min(fps, 30)  # Cap at 30 FPS for better performance
    frame_delay = 1.0 / target_fps
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video reached. Press 'r' to restart or 'q' to quit.")
                key = cv2.waitKey(0) & 0xFF
                if key == ord('r'):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_count = 0
                    continue
                elif key == ord('q'):
                    break
                else:
                    continue
            
            # Resize frame for consistent processing (optional)
            if width > 640:
                scale = 640 / width
                new_width = 640
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Process frame with lane detection
            start_time = time.time()
            processed_frame = lane_detector.process_frame(frame)
            processing_time = time.time() - start_time
            
            nav_cmd = lane_detector.get_navigation_command()
            
            
    
            
    
            
            # Processing time info
            if processing_time > 0:
                fps_text = f"Processing: {1/processing_time:.1f} FPS"
            else:
                fps_text = "Processing: >1000 FPS"
                cv2.putText(processed_frame, fps_text, (processed_frame.shape[1] - 200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # Print console output every 0.5 seconds
        
            
            cv2.imshow('Video Lane Detection', processed_frame)
            
            # Frame rate control
            elapsed = time.time() - start_time
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)
        
        else:
            # Paused - just wait for key press
            cv2.waitKey(30)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and not paused:
            filename = f"lane_detection_frame_{frame_count}_{current_time_sec:.1f}s.jpg"
            cv2.imwrite(filename, processed_frame)
            print(f"Saved frame: {filename}")
        elif key == ord('p'):
            paused = not paused
            print("Video paused" if paused else "Video resumed")
        elif key == ord('r'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0
            paused = False
            print("Video restarted")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Video lane detection completed for: {os.path.basename(video_path)}")

if __name__ == "__main__":
    main()