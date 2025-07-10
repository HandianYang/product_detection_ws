import rospy
from detection_msgs.msg import DetectedObject, DetectedObjectArray
from geometry_msgs.msg import Point
import std_msgs.msg
from sensor_msgs.msg import Image, CameraInfo
import tf2_ros
import tf2_geometry_msgs

import cv2
from cv_bridge import CvBridge
import numpy as np
from pathlib import Path
import pyrealsense2 as rs
from ultralytics import YOLO
import yaml


class YoloInference():
    DEFAULT_CONFIDENCE_THRESHOLD = 0.7
    DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().parent.parent.parent / "config/class_list.yaml")
    DEFAULT_WEIGHT_PATH = str(Path(__file__).resolve().parent.parent.parent / "weight/yolo11_v1.pt")

    # The transformation from camera_link to tm_tip_link is defined as:
    #   x: from centerline to left imager (D435/D435i: 17.5 mm = 0.0175 m)
    #   y: by manual measurement (6 cm = 0.06 m)
    #   z: from /tm_tip_link origin to front glass cover by manual measurement (-9 cm = -0.09 m)
    #    + from front glass cover to depth start point (D435/D435i: 4.2 mm = 0.0042 m)
    #    = -0.0858 m
    CAMERA_LINK_TO_TIP_LINK_T = np.array([0.0175, 0.06, -0.0858])

    def __init__(self):
        self.__confidence_threshold = rospy.get_param("~confidence_threshold", YoloInference.DEFAULT_CONFIDENCE_THRESHOLD)
        self.__config_path = rospy.get_param("~config_path", YoloInference.DEFAULT_CONFIG_PATH)
        self.__display_enabled = rospy.get_param("~display_enabled", False)
        self.__weight_path = rospy.get_param("~weight_path", YoloInference.DEFAULT_WEIGHT_PATH)

        self.__model = YOLO(self.__weight_path)
        self.__class_names = self.__load_class_names_from_yaml()

        self.__bridge = CvBridge()
        self.__color_image = Image()
        self.__depth_image = Image()
        self.__camera_intrinsics = rs.intrinsics()

        self.__tf_buffer = tf2_ros.Buffer()
        self.__tf_listener = tf2_ros.TransformListener(self.__tf_buffer)

        self.__color_image_subscriber = rospy.Subscriber("/camera/color/image_raw", Image, self.__camera_color_image_callback)
        self.__depth_image_subscriber = rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.__camera_depth_image_callback)
        self.__camera_info_subscriber = rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.__camera_info_callback)

        self.__detected_objects_publisher = rospy.Publisher('/yolo/detected_objects', DetectedObjectArray, queue_size=1)
    
    # === Public Methods ===
    def get_inference_results(self) -> None:
        try:
            if not self.__color_image.data or self.__color_image.height == 0 or self.__color_image.width == 0:
                rospy.logwarn("Color image data is empty or has invalid dimensions. Skipping inference.")
                return
            bgr_image = np.frombuffer(self.__color_image.data, dtype=np.uint8).reshape(self.__color_image.height, self.__color_image.width, -1)
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            predictions = self.__model.predict(
                source=rgb_image,
                conf=self.__confidence_threshold,
                show=False,
                save=False,
                verbose=False)[0]

            if self.__display_enabled:
                annotated_img = predictions.plot()
                cv2.imshow("YOLOv11 Detection", annotated_img)
                key = cv2.waitKey(1)
                if key == 27:  # ESC key to close window and shutdown the node
                    cv2.destroyAllWindows()
                    rospy.signal_shutdown("ESC pressed")
        except ValueError as e:
            rospy.logerr(f"Error processing color image: {e}")
            return

        detected_objects = DetectedObjectArray()
        for prediction in predictions.boxes:
            detected_object = DetectedObject()
            detected_object.label = self.__class_names[int(prediction.cls[0].item())]
            detected_object.confidence = prediction.conf[0].item()
            detected_object.centroid = self.__get_centroid_wrt_base_link(prediction.xyxy[0].tolist())
            if detected_object.centroid is None:
                rospy.logwarn(f"Centroid for {detected_object.label} could not be computed. Skipping this object.")
                continue
            detected_objects.objects.append(detected_object)
        self.__detected_objects_publisher.publish(detected_objects)

    # === Private Methods (initialization) ===
    def __load_class_names_from_yaml(self) -> list:
        with open(self.__config_path, 'r') as f:
            data = yaml.safe_load(f)
        return data['class_names']

    # === Private Methods (supports get_inference_results()) ===
    def __get_centroid_wrt_base_link(self, bbox: list) -> Point:
        centroid_wrt_camera_link = self.__estimate_centroid_from_bbox(bbox)
        if centroid_wrt_camera_link is None:
            return None
        centroid_wrt_tip_link = self.__transfrom_camera_link_to_tip_link(centroid_wrt_camera_link)
        centroid_wrt_base_link = self.__transform_tip_link_to_base_link(centroid_wrt_tip_link)
        if centroid_wrt_base_link is None:
            return None
        return centroid_wrt_base_link
    
    def __estimate_centroid_from_bbox(self, bbox: list) -> Point:
        u_min, v_min, u_max, v_max = map(int, bbox)
        u_center = int((u_min + u_max) / 2)
        v_center = int((v_min + v_max) / 2)
        depth = self.__compute_depth_around_center(u_center=u_center, v_center=v_center)
        if depth is None:
            return None

        point = rs.rs2_deproject_pixel_to_point(self.__camera_intrinsics, [u_center, v_center], depth)
        centroid = Point()
        centroid.x = point[0]
        centroid.y = point[1]
        centroid.z = point[2]
        return centroid

    def __compute_depth_around_center(self, u_center: int, v_center: int, delta: int = 20) -> float:
        u_min = max(0, u_center - delta)
        u_max = min(self.__depth_image.shape[1], u_center + delta)
        v_min = max(0, v_center - delta)
        v_max = min(self.__depth_image.shape[0], v_center + delta)
        patch = self.__depth_image[u_min:u_max, v_min:v_max].flatten()
        valid = patch[(patch > 0.2) & (patch < 2)]

        if len(valid) == 0:
            return None
        else:
            return float(np.median(valid))

    def __transfrom_camera_link_to_tip_link(self, centroid_wrt_camera_link: Point) -> Point:
        centroid_wrt_tip_link = Point()
        centroid_wrt_tip_link.x = centroid_wrt_camera_link.x - YoloInference.CAMERA_LINK_TO_TIP_LINK_T[0]
        centroid_wrt_tip_link.y = centroid_wrt_camera_link.y - YoloInference.CAMERA_LINK_TO_TIP_LINK_T[1]
        centroid_wrt_tip_link.z = centroid_wrt_camera_link.z - YoloInference.CAMERA_LINK_TO_TIP_LINK_T[2]
        return centroid_wrt_tip_link
    
    def __transform_tip_link_to_base_link(self, centroid_wrt_tip_link: Point) -> Point:
        try:
            if not self.__tf_buffer.can_transform('base_link', 'tm_tip_link', rospy.Time(0), rospy.Duration(1.0)):
                rospy.logwarn("Cannot transform from /tm_tip_link to /base_link")
                return None
            point_stamped = tf2_geometry_msgs.PointStamped()
            point_stamped.header.frame_id = 'tm_tip_link'
            point_stamped.header.stamp = rospy.Time(0)
            point_stamped.point = centroid_wrt_tip_link
            transformed_point_stamped = self.__tf_buffer.transform(point_stamped, 'base_link', rospy.Duration(1.0))
            return transformed_point_stamped.point
        except Exception as e:
            rospy.logerr(f"TF2 transform error: {e}")
            return None

    # === Private Methods (callbacks) ===
    def __camera_color_image_callback(self, color_image: Image) -> None:
        if not color_image.data:
            rospy.logwarn("No data received for color image. Skipping this frame.")
            return
        self.__color_image = color_image
    
    def __camera_depth_image_callback(self, depth_image: Image) -> None:
        try:
            # Convert ROS image to OpenCV (16UC1 or 32FC1 depending on camera)
            self.__depth_image = self.__bridge.imgmsg_to_cv2(depth_image, desired_encoding="passthrough")
        except Exception as e:
            rospy.logerr(f"Depth image conversion error: {e}")
            return

        # Convert to meters if needed
        if self.__depth_image.dtype == np.uint16:
            self.__depth_image = self.__depth_image.astype(np.float32) / 1000.0

    def __camera_info_callback(self, camera_info: CameraInfo) -> None:
        self.__camera_intrinsics.width = camera_info.width
        self.__camera_intrinsics.height = camera_info.height
        self.__camera_intrinsics.ppx = camera_info.K[2]
        self.__camera_intrinsics.ppy = camera_info.K[5]
        self.__camera_intrinsics.fx = camera_info.K[0]
        self.__camera_intrinsics.fy = camera_info.K[4]
        self.__camera_intrinsics.model = rs.distortion.none
        self.__camera_intrinsics.coeffs = [i for i in camera_info.D]