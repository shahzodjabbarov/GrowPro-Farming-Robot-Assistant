project:
  name: GrowPro – Intelligent Farming Robot Assistant
  description: >
    GrowPro is an autonomous farming robot designed to assist with crop monitoring,
    navigation, and agricultural analysis using computer vision, deep learning,
    and sensor-based systems.

  domain: Precision Agriculture / Robotics / AI
  status: Completed Prototype
  type: Academic + Research Project

capabilities:
  operation_modes:
    - Autonomous navigation with AI-based crop monitoring
    - Manual remote control with live video streaming
    - Person-following mode using visual tracking

  vision_ai:
    - Crop disease detection
    - Ripeness classification
    - Weed detection
    - Crop counting (yield estimation)
    - Soil dry-spot recognition

  navigation:
    - Outdoor camera-based lane detection
    - Indoor/greenhouse line following
    - Ultrasonic obstacle avoidance
    - Smooth proportional steering control

system_architecture:
  overview: >
    The system is divided into robot-side execution, server-side intelligence,
    and user-side interaction, communicating via socket-based networking.

  data_flow:
    - User Interface (Laptop)
    - Socket Communication
    - Server-Side AI & CV Processing
    - Video & Sensor Streaming
    - Robot-Side Motor & Sensor Control

hardware:
  computing:
    - Raspberry Pi 4 Model B (8GB RAM, active cooling)

  vision:
    - Dual high-resolution cameras (navigation + crop analysis)

  sensors:
    - Ultrasonic distance sensors
    - IR line-following sensors
    - Soil humidity sensors

  actuation:
    - Four-wheel drive motors
    - Servo motors for sampling tasks

  chassis:
    - Yahboom Raspbot (prototype)
    - Planned custom motor driver board

software:
  languages:
    - Python

  frameworks_libraries:
    - OpenCV
    - YOLOv8 (Ultralytics)
    - Pygame

  communication:
    - Socket-based video streaming
    - Command-based remote control

  operating_systems:
    - Raspberry Pi OS
    - Linux

  external_services:
    - Weather API integration for farming alerts

repository_structure:
  root:
    - README.md
    - media/

  robot_side:
    description: Code executed directly on the robot hardware
    modules:
      - motor_control
      - navigation
      - remote_net

  server_side:
    description: AI, computer vision, and high-level control logic
    modules:
      - control
      - cv_models
      - detection

  user_side:
    description: User-facing applications and interfaces
    modules:
      - application
      - manual_control
      - navigation_app

computer_vision_pipeline:
  steps:
    - Image acquisition
    - CLAHE contrast enhancement
    - Gaussian blur (5x5)
    - Grayscale conversion
    - ROI masking (approx. 60% background removal)
    - Adaptive Canny edge detection
    - Morphological cleaning (3x3 kernel)
    - Lane center calculation
    - Motor command generation

yolo_detection:
  model: YOLOv8
  dataset:
    crops:
      - Pumpkin (A, B)
      - Salad (A, B)
      - Strawberry (A, B)

  configuration:
    confidence_threshold: 0.3 – 0.65
    optimization:
      - Frame skipping
      - Resolution scaling
      - Context-aware class filtering

communication:
  video_streaming:
    protocol: Socket + JPEG compression
    port: 8491

  control_commands:
    protocol: Socket-based
    port: 8490

  performance:
    - Low-latency streaming
    - Frame dropping for real-time stability

user_interface:
  framework: Pygame
  features:
    - Field selection
    - Crop selection
    - Mode switching (Auto / Manual / Follow)
    - Live annotated camera feed
    - Visual farm map
    - Issue marker placement
    - Weather dashboard

media:
  policy: >
    Large videos and datasets are not stored in GitHub.
    All media is hosted externally and linked from the media directory.

  contents:
    - Build process videos
    - Robot demonstration videos
    - Seminar and presentation videos
    - Posters and certificates

impact:
  benefits:
    - Approximately 70% reduction in manual inspection time
    - Earlier detection of crop diseases
    - Reduced water waste through precise monitoring
    - Data-driven farming decisions

future_work:
  hardware:
    - Custom motor driver board
    - 360-degree camera coverage
    - LiDAR integration
    - Extended battery capacity

  software:
    - Mobile companion application
    - Cloud-based analytics dashboard
    - Multi-robot coordination
    - Automated irrigation integration

notes:
  - This repository contains the full software architecture of the GrowPro system.
  - Datasets and trained models are referenced externally.
