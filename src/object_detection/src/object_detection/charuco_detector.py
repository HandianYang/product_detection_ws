#!/usr/bin/env python3
from pathlib import Path

import cv2
import cv2.aruco as aruco
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TransformStamped
import numpy as np
import rospy
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image, CameraInfo
import tf2_ros
import yaml

class CharucoDetector:
    def __init__(self):
        # ArUco marker dictionary
        self.__aruco_dict_id = aruco.DICT_6X6_250  # contains ID 0~249
        self.__dictionary = aruco.getPredefinedDictionary(self.__aruco_dict_id)

        # ChArUco board parameters
        self.__num_squares_x = 10
        self.__num_squares_y = 7
        self.__square_length = 0.015
        self.__marker_length = 0.011
        self.__board = aruco.CharucoBoard(
            (self.__num_squares_x, self.__num_squares_y), 
            self.__square_length, self.__marker_length, 
            self.__dictionary
        )

        # ChArUco detector parameters
        self.__detector_params = aruco.DetectorParameters()
        self.__charuco_detector = aruco.CharucoDetector(
            self.__board, 
            charucoParams=aruco.CharucoParameters(), 
            detectorParams=self.__detector_params
        )

        self.__tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.__bridge = CvBridge()
        try:
            yaml_path = str(Path(__file__).resolve().parent.parent.parent / "config/realsense_intrinsics.yaml")
            camera_info = self.__load_camera_info_from_yaml(yaml_path)
            self.__camera_matrix = np.array(camera_info.K).reshape(3, 3)
            self.__dist_coeffs = np.array(camera_info.D)
            rospy.loginfo(f"Successfully loaded calibration from {yaml_path}")
        except Exception as e:
            rospy.logerr(f"Failed to load yaml: {e}")
            self.__camera_matrix = None
            self.__dist_coeffs = None
        
        self.__camera_image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.__image_callback)
        self.__charuco_pose_pub = rospy.Publisher("/charuco/pose", PoseStamped, queue_size=10)
        self.__charuco_image_pub = rospy.Publisher("/charuco/result", Image, queue_size=1)

    def __load_camera_info_from_yaml(self, yaml_path: str) -> CameraInfo:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        camera_info_msg = CameraInfo()    
        camera_info_msg.width = data["image_width"]
        camera_info_msg.height = data["image_height"]
        camera_info_msg.distortion_model = data["distortion_model"]
        camera_info_msg.D = data["distortion_coefficients"]["data"]
        camera_info_msg.K = data["camera_matrix"]["data"]
        camera_info_msg.R = data["rectification_matrix"]["data"]
        camera_info_msg.P = data["projection_matrix"]["data"]
        return camera_info_msg

    def __image_callback(self, msg: Image) -> None:
        if self.__camera_matrix is None:
            return

        try:
            cv_image = self.__bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(e)
            return

        charuco_corners, charuco_ids, _, _ = self.__charuco_detector.detectBoard(cv_image)
        if charuco_corners is not None and len(charuco_corners) >= 6:
            obj_points, img_points = self.__board.matchImagePoints(charuco_corners, charuco_ids)
            valid, rvec, tvec = cv2.solvePnP(
                obj_points, 
                img_points, 
                self.__camera_matrix, 
                self.__dist_coeffs
            )

            if valid:
                aruco.drawDetectedCornersCharuco(cv_image, charuco_corners, charuco_ids)
                cv2.drawFrameAxes(cv_image, self.__camera_matrix, self.__dist_coeffs, rvec, tvec, 0.1)
                
                self.__publish_pose(rvec, tvec, msg.header)

        self.__charuco_image_pub.publish(self.__bridge.cv2_to_imgmsg(cv_image, "bgr8"))

    def __publish_pose(self, rvec, tvec, header):
        pose_msg = PoseStamped()
        pose_msg.header = header
        pose_msg.header.frame_id = "camera_color_optical_frame"
        
        pose_msg.pose.position.x = tvec[0][0]
        pose_msg.pose.position.y = tvec[1][0]
        pose_msg.pose.position.z = tvec[2][0]
        
        rot_mat, _ = cv2.Rodrigues(rvec)
        quat = R.from_matrix(rot_mat).as_quat()        
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]
        
        self.__charuco_pose_pub.publish(pose_msg)

        t = TransformStamped()
        t.header.stamp = header.stamp
        t.header.frame_id = "camera_color_optical_frame"
        t.child_frame_id = "charuco_board_frame"
        t.transform.translation = pose_msg.pose.position
        t.transform.rotation = pose_msg.pose.orientation
        
        self.__tf_broadcaster.sendTransform(t)

