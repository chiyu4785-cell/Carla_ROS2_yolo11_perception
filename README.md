# Carla_ROS2_yolo11_perception

## Overview
This project implements a perception module in CARLA.
A front-view RGB camera publishes images via ROS2, and a YOLO model is used for real-time object detection.

## Features
- CARLA front-view camera
- ROS2 image publishing
- YOLO (PyTorch) object detection
- OpenCV visualization

## System Architecture
CARLA Camera → ROS2 → YOLO → OpenCV

## How to Run
1. Start CARLA server
2. Run the perception script

## ROS2 Topics
- /camera/image_raw
- 
## System Dependencies
- Ubuntu 22.04
- ROS2 Humble
  - Install Python packages: rclpy, cv_bridge, sensor_msgs via apt
- CARLA 0.9.16
- Python 3.10
