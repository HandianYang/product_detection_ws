#!/usr/bin/env python3

import cv2
import cv2.aruco as aruco
from cv_bridge import CvBridge
import numpy as np
import rospy
from sensor_msgs.msg import Image
import yaml

class IntrinsicsCalibrator:
    def __init__(self):
        self.__squares_x = 10
        self.__squares_y = 7
        self.__square_length = 0.015
        self.__marker_length = 0.011
        self.__aruco_dict_id = cv2.aruco.DICT_6X6_250
        
        self.__dictionary = cv2.aruco.getPredefinedDictionary(self.__aruco_dict_id)
        self.__charuco_board = cv2.aruco.CharucoBoard(
            (self.__squares_x, self.__squares_y), 
            self.__square_length, 
            self.__marker_length, 
            self.__dictionary
        )
        self.__charuco_detector = cv2.aruco.CharucoDetector(self.__charuco_board)
        
        self.__all_charuco_corners = []
        self.__all_charuco_ids = []
        self.__image_size = None
        self.__captured_count = 0
        
        self.__bridge = CvBridge()
        self.__image_subscriber = rospy.Subscriber("/camera/color/image_raw", Image, self.__image_callback)
        self.__latest_image = None
        
        print("=== ChArUco Calibration Tool ===")
        print("1. Move the board around.")
        print("2. Press 'c' to CAPTURE a frame.")
        print("3. Press 'q' to FINISH and CALIBRATE.")

    # === Public Methods ===
    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.__latest_image is None:
                continue
                
            display_img = self.__latest_image.copy()
            charuco_corners, charuco_ids, _, _ = self.__charuco_detector.detectBoard(self.__latest_image)
            
            if charuco_corners is not None and len(charuco_corners) > 4:
                cv2.aruco.drawDetectedCornersCharuco(display_img, charuco_corners, charuco_ids)
                cv2.putText(display_img, "Ready to Capture", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(display_img, "Show Board...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(display_img, f"Captured: {self.__captured_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.imshow("Calibration", display_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                if charuco_corners is not None and len(charuco_corners) > 4:
                    self.__all_charuco_corners.append(charuco_corners)
                    self.__all_charuco_ids.append(charuco_ids)
                    self.__image_size = self.__latest_image.shape[:2] # (h, w)
                    self.__captured_count += 1
                    print(f"Captured frame {self.__captured_count}")
                else:
                    print("Board not detected well, skip.")
            elif key == ord('q'):
                if self.__captured_count > 10:
                    self.__calibrate()
                    break
                else:
                    print("Need at least 10 frames!")

            rate.sleep()
        
        cv2.destroyAllWindows()

    # === Private Methods (callbacks) ===
    def __image_callback(self, msg):
        try:
            self.__latest_image = self.__bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(e)

    # === Private Methods (supports run()) ===
    def __calibrate(self):
        print("\nStarting Calibration... This may take a moment.")
        
        all_object_points = []
        all_image_points = []
        for c_corners, c_ids in zip(self.__all_charuco_corners, self.__all_charuco_ids):
            if c_corners is None or len(c_corners) < 4:
                continue

            object_points, image_points = self.__charuco_board.matchImagePoints(c_corners, c_ids)
            if object_points is not None and len(object_points) >= 4:
                all_object_points.append(object_points)
                all_image_points.append(image_points)

        if len(all_object_points) < 10:
            print("Not enough valid frames for calibration! (Need > 10)")
            return
        print(f"Calibrating with {len(all_object_points)} valid frames...")
        
        reprojection_error, camera_matrix, distortion_coefficient, _, _ = cv2.calibrateCamera(
            objectPoints=all_object_points, 
            imagePoints=all_image_points, 
            imageSize=self.__image_size, 
            cameraMatrix=None,
            distCoeffs=None
        )
        
        print("\n=== Calibration Result ===")
        print(f"Reprojection Error: {reprojection_error:.4f} (Should be < 1.0, ideally < 0.5)")
        print("Camera Matrix (K):\n", camera_matrix)
        print("Distortion Coefficients (D):\n", distortion_coefficient)
        self.__save_yaml(camera_matrix, distortion_coefficient, reprojection_error)

    def __save_yaml(self, camera_matrix, distortion_coefficient, error):
        data = {
            'image_width': self.__image_size[1],
            'image_height': self.__image_size[0],
            'camera_name': 'camera',
            'camera_matrix': {
                'rows': 3,
                'cols': 3,
                'data': camera_matrix.flatten().tolist()
            },
            'distortion_model': 'plumb_bob',
            'distortion_coefficients': {
                'rows': 1,
                'cols': 5,
                'data': distortion_coefficient.flatten().tolist()
            },
            'rectification_matrix': {
                'rows': 3,
                'cols': 3,
                'data': np.eye(3).flatten().tolist()
            },
            'projection_matrix': {
                'rows': 3,
                'cols': 4,
                'data': np.hstack((camera_matrix, np.zeros((3, 1)))).flatten().tolist()
            }
        }
        
        filename = f"realsense_intrinsics_{error:.3f}.yaml"
        with open(filename, 'w') as f:
            yaml.dump(data, f)
        print(f"\nSaved calibration to {filename}")

