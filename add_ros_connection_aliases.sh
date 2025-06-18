#!/bin/bash

echo "" >> ~/.bashrc
echo "# Default ROS connection setting" >> ~/.bashrc
echo "export ROS_HOSTNAME=localhost" >> ~/.bashrc
echo "export ROS_MASTER_URI=http://localhost:11311" >> ~/.bashrc
echo "export ROS_IP=localhost" >> ~/.bashrc

echo "" >> ~/.bashrc
echo "# Switch to robot master (Ethernet cable connection)" >> ~/.bashrc
echo "alias robot='export ROS_HOSTNAME=192.168.10.2 && export ROS_MASTER_URI=http://192.168.10.1:11311 && export ROS_IP=192.168.10.1'" >> ~/.bashrc
echo "# Switch to local master" >> ~/.bashrc
echo "alias local='export ROS_HOSTNAME=localhost && export ROS_MASTER_URI=http://localhost:11311 && export ROS_IP=localhost'" >> ~/.bashrc

source ~/.bashrc