# Product Detection ROS Workspace

## Prerequisites

+ Hardware:
  - GPU: any CUDA 12.1-capable (RTX 3060 Ti or better)
  - RealSense camera: D435/D435i
+ Software:
  - GPU driver: >= 525.60
  - Docker: >= 20.10
  - NVIDIA Container Toolkit (`nvidia-docker2`)


## Setup

### 1. Enter Docker container

```bash
source docker_run.sh
```

### 2. Build ROS workspace

```bash
catkin_make

# ...or type the alias:
cm
```

### 3. Link to `devel/` sources
```bash
source devel/setup.bash

# ...or type the alias:
sd
```

### 4. Connect to robot (ROS master)
```bash
source switch_to_robot_ros_connection.sh
```


## Demonstration

### Realsense point cloud demo

```bash
roslaunch detection_examples realsense_pointcloud_demo.launch
```

### YOLO inference feature

**[Note] Remember to connect Realsense D435/D435i camera first!**

#### Boundingbox centroid estimation

```bash
roslaunch detection_examples yolo_inference_bbox_estimator.launch
```

#### Pointcloud centroid estimation

```bash
roslaunch detection_examples yolo_inference_pointcloud_estimator.launch
```
