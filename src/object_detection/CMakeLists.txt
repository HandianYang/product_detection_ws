################################################################################
# Set minimum required version of cmake, project name and compile options
################################################################################

cmake_minimum_required(VERSION 3.5)
project(object_detection)


################################################################################
# Find catkin packages and libraries for catkin and system dependencies
################################################################################
find_package(catkin REQUIRED COMPONENTS
  # geometry_msgs
  rospy
  sensor_msgs
  # std_msgs
)
find_package(OpenCV REQUIRED)

################################################################################
# Setup for python modules and scripts
################################################################################
catkin_python_setup()

################################################################################
# Declare ROS messages, services and actions
################################################################################

################################################################################
# Declare ROS dynamic reconfigure parameters
################################################################################


################################################################################
# Declare catkin specific configuration to be passed to dependent projects
################################################################################

catkin_package(
  CATKIN_DEPENDS
    # geometry_msgs
    rospy
    sensor_msgs
    # std_msgs
)

################################################################################
# Build
################################################################################

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

################################################################################
# Install
################################################################################

# Install Python scripts
# catkin_install_python(PROGRAMS
#     tests/test_yolo_inference.py
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

# Install Python modules
# install(DIRECTORY
#     scripts/object_detection
#   DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
# )

# # Install launch files
# install(DIRECTORY launch
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
# )


################################################################################
# Test
################################################################################