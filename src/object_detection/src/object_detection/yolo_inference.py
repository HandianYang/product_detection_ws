from enum import Enum
from pathlib import Path

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
import numpy as np
import pyrealsense2 as rs
import rospy
from sensor_msgs.msg import CameraInfo, Image
from sklearn.cluster import KMeans
from ultralytics import YOLO
from visualization_msgs.msg import Marker, MarkerArray
import yaml

from detection_msgs.msg import DetectedObject, DetectedObjectArray

class CentroidEstimator(Enum):
    BBOX = 1
    POINTCLOUD = 2

class YoloInference:
    DEFAULT_CONFIDENCE_THRESHOLD = 0.7
    DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().parent.parent.parent / "config/class_list.yaml")
    DEFAULT_INTRINSICS_PATH = str(Path(__file__).resolve().parent.parent.parent / "config/realsense_intrinsics_20251211.yaml")
    DEFAULT_WEIGHT_PATH = str(Path(__file__).resolve().parent.parent.parent / "weight/yolo11_v1.pt")

    def __init__(self, centroid_estimator: CentroidEstimator = CentroidEstimator.BBOX) -> None:
        # parameters
        self.__confidence_threshold = rospy.get_param("~confidence_threshold", YoloInference.DEFAULT_CONFIDENCE_THRESHOLD)
        self.__display_enabled = rospy.get_param("~display_enabled", False)
        self.__weight_path = rospy.get_param("~weight_path", YoloInference.DEFAULT_WEIGHT_PATH)

        # YOLO
        self.__model = YOLO(self.__weight_path)
        self.__class_names = self.__load_class_names_from_yaml()
        self.__predictions = None
        self.__centroid_estimator = centroid_estimator

        # camera
        self.__bridge = CvBridge()
        self.__color_image = Image()
        self.__depth_image = Image()
        self.__camera_intrinsics = rs.intrinsics()

        # subscribers
        self.__color_image_subscriber = rospy.Subscriber("/camera/color/image_raw", Image, self.__camera_color_image_callback)
        self.__depth_image_subscriber = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.__camera_depth_image_callback)
        self.__camera_info_subscriber = rospy.Subscriber("/camera/aligned_depth_to_color/camera_info", CameraInfo, self.__camera_info_callback)

        # publishers
        self.__detected_objects_publisher = rospy.Publisher("/yolo/detected_objects", DetectedObjectArray, queue_size=1)
        self.__detected_objects_marker_publisher = rospy.Publisher("/yolo/detected_objects_marker", MarkerArray, queue_size=1)
    
    # === Public Methods ===
    def run_inference(self) -> None:
        """    
        Run the YOLO inference process continuously until shutdown.

        The loop rate is fixed at 10 Hz as recommended frequency, because each loop
        takes roughly 0.01~0.06 seconds.

        Examples:

            yolo_inference = YoloInference()
            yolo_inference.run_inference()
        """
        default_rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.run_inference_once()
            default_rate.sleep()

    def run_inference_once(self) -> None:
        """    
        Run the YOLO inference process once.

        It's useful for developers to understand the standard process of inference
        and how to handle the results. Furthermore, you can specify loop rate on 
        your own. Note that each loop takes roughly 0.01~0.06 seconds, so it's
        recommended to set the frequency below 10 Hz.

        Examples:

            while not rospy.is_shutdown():
                yolo_inference.run_inference_once()
                rospy.Rate(5).sleep()
        """
        self.__calculate_inference_results()
        if self.__predictions is None:
            return
        self.__plot_inference_results()
        
        detected_objects = self.__get_detected_objects()
        self.__publish_detected_objects(detected_objects)

        detected_objects_marker = self.__get_detected_objects_marker(detected_objects)
        self.__publish_detected_objects_marker(detected_objects_marker)

    # === Private Methods (initialization) ===
    def __load_class_names_from_yaml(self) -> list:
        with open(self.DEFAULT_CONFIG_PATH, 'r') as f:
            data = yaml.safe_load(f)
        return data['class_names']

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

        # Convert to meters
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

    # === Private Methods (supports run_inference_once()) ===
    def __calculate_inference_results(self) -> None:
        try:
            if not self.__color_image.data or self.__color_image.height == 0 or self.__color_image.width == 0:
                rospy.logwarn("Color image data is empty or has invalid dimensions. Skipping inference.")
                return
            bgr_image = np.frombuffer(self.__color_image.data, dtype=np.uint8).reshape(self.__color_image.height, self.__color_image.width, -1)
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            self.__predictions = self.__model.predict(
                source=rgb_image,
                conf=self.__confidence_threshold,
                show=False,
                save=False,
                verbose=False)[0]            
        except ValueError as e:
            rospy.logerr(f"Error processing color image: {e}")
            return
    
    def __plot_inference_results(self) -> None:
        if not self.__display_enabled or self.__predictions is None:
            return
        annotated_img = self.__predictions.plot()
        cv2.imshow("YOLOv11 Detection", annotated_img)
        key = cv2.waitKey(1)
        if key == 27:  # ESC key to close window and shutdown the node
            cv2.destroyAllWindows()
            rospy.signal_shutdown("ESC pressed")
    
    def __get_detected_objects(self) -> DetectedObjectArray:
        detected_objects = DetectedObjectArray()
        for prediction in self.__predictions.boxes:
            detected_object = DetectedObject()
            detected_object.label = self.__class_names[int(prediction.cls[0].item())]
            detected_object.frame = 'camera_color_optical_frame'
            detected_object.confidence = prediction.conf[0].item()
            detected_object.centroid = self.__get_centroid_wrt_color_optical_frame(prediction.xyxy[0].tolist())
            if detected_object.centroid is None:
                continue
            detected_objects.objects.append(detected_object)
        return detected_objects

    def __get_centroid_wrt_color_optical_frame(self, bbox: list) -> Point:
        if self.__centroid_estimator == CentroidEstimator.BBOX:
            centroid_wrt_camera_link = self.__estimate_centroid_with_bbox_center(bbox)
        elif self.__centroid_estimator == CentroidEstimator.POINTCLOUD:
            centroid_wrt_camera_link = self.__estimate_centroid_with_pointcloud_clustering(bbox)
        
        if centroid_wrt_camera_link is None:
            return None
        return centroid_wrt_camera_link

    def __estimate_centroid_with_bbox_center(self, bbox: list) -> Point:
        """
        Estimate the centroid of an object according to the depth value of the
        center pixel of the bounding box.

        This method computes median of depth value of a small area around the
        center pixel of the bounding box, and deprojects the centroid using 
        center pixel and the depth value.

        Normally, this method takes less time, but it cannot avoid the problem
        that the center pixel of the partial-hidden object may be one pixel of
        the object in front.

        Args:
            bbox (List[Float]): A list containing the coordinates of the
                bounding box in the format [u_min, v_min, u_max, v_max].
        
        Returns:
            Point (geometry_msgs/Point): The estimated centroid of the object 
        """
        center_depth = self.__compute_depth_around_center(bbox)
        if center_depth is None:
            return None
        
        u_min, v_min, u_max, v_max = map(int, bbox)
        u_center = int((u_min + u_max) / 2)
        v_center = int((v_min + v_max) / 2)
        point = rs.rs2_deproject_pixel_to_point(self.__camera_intrinsics, [u_center, v_center], center_depth)
        centroid = Point()
        centroid.x = point[0]
        centroid.y = point[1]
        centroid.z = point[2]
        return centroid

    def __compute_depth_around_center(self, bbox: list, delta: int = 5) -> float:
        u_min, v_min, u_max, v_max = map(int, bbox)
        u_center = int((u_min + u_max) / 2)
        v_center = int((v_min + v_max) / 2)
        center_depth = self.__depth_image[v_center, u_center]
        if 0.2 < center_depth < 3:
            return center_depth

        u_min = max(0, u_center - delta)
        u_max = min(self.__depth_image.shape[1], u_center + delta)
        v_min = max(0, v_center - delta)
        v_max = min(self.__depth_image.shape[0], v_center + delta)
        patch = self.__depth_image[v_min:v_max, u_min:u_max].flatten()
        valid = patch[(patch > 0.2) & (patch < 3)]
        
        if len(valid) == 0:
            return None
        return float(np.median(valid))
    
    def __estimate_centroid_with_pointcloud_clustering(self, bbox: list) -> Point:
        """
        Estimate the centroid of an object according to the depth values of the
        point cloud.

        This method computes the major point cloud cluster as the detected
        object, takes the smallest 5% depth value as the center pixel
        depth, and deprojects the centroid using center pixel and the specific
        depth value.

        Although this method takes longer time, it can (ideally) solve the
        problem mentioned in `__estimate_centroid_with_bbox_center()`

        Args:
            bbox (List[Float]): A list containing the coordinates of the
                bounding box in the format [u_min, v_min, u_max, v_max].
        
        Returns:
            Point (geometry_msgs/Point): The estimated centroid of the object 
        """
        sampled_pixels = self.__get_sampled_filtered_points(bbox)
        major_group_pixels = self.__get_major_kmean_cluster(bbox, sampled_pixels)
        if major_group_pixels is None:
            return None
        surface_depth = np.percentile(major_group_pixels, 5)
        return self.__estimate_centroid_with_specific_depth(bbox, surface_depth)
    
    def __get_sampled_filtered_points(self, bbox: list, sample: int = 10) -> np.array:
        u_min, v_min, u_max, v_max = map(int, bbox)
        roi = self.__depth_image[v_min:v_max, u_min:u_max].copy()
        valid_mask = (roi > 0.2) & (roi < 3)
        valid_pixels = roi[valid_mask]
        sample_pixels = valid_pixels[::sample] if len(valid_pixels) > 500 else valid_pixels
        return sample_pixels
    
    def __get_major_kmean_cluster(self, bbox: list, sampled_pixels: np.array) -> np.array:
        try:
            kmeans = KMeans(n_clusters=2, n_init=1, max_iter=10).fit(sampled_pixels.reshape(-1, 1))
        except Exception as e:
            return None

        labels = kmeans.labels_
        centers = kmeans.cluster_centers_.flatten()
        count_group_0 = np.sum(labels == 0)
        count_group_1 = np.sum(labels == 1)
        if count_group_0 > count_group_1:
            major_group_label = 0
            major_center = centers[0]
        else:
            major_group_label = 1
            major_center = centers[1]        
        major_group_pixels = sampled_pixels[labels == major_group_label]

        ## TODO: Test the functionality
        center_depth = self.__compute_depth_around_center(bbox)
        if center_depth is None:
            return None
        depth_diff = center_depth - major_center
        if abs(depth_diff) > 0.05:
            if depth_diff < -0.05:
                print("Rejected: 中心被遮擋 (Center on Occluder)")
            else:
                print("Rejected: 中心落在背景 (Center on Background)")
            return None
        return major_group_pixels

    def __estimate_centroid_with_specific_depth(self, bbox, depth):
        u_min, v_min, u_max, v_max = map(int, bbox)
        u_center = int((u_min + u_max) / 2)
        v_center = int((v_min + v_max) / 2)
        point = rs.rs2_deproject_pixel_to_point(self.__camera_intrinsics, [u_center, v_center], depth)
        centroid = Point()
        centroid.x = point[0]
        centroid.y = point[1]
        centroid.z = point[2]
        return centroid

    def __publish_detected_objects(self, detected_objects: DetectedObjectArray) -> None:
        self.__detected_objects_publisher.publish(detected_objects)

    def __get_detected_objects_marker(self, detected_objects: DetectedObjectArray) -> MarkerArray:
        marker_array = MarkerArray()
        for i, detected_object in enumerate(detected_objects.objects):
            marker = Marker()
            marker.header.frame_id = "camera_color_optical_frame"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "detected_objects"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = detected_object.centroid
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.01
            marker.scale.y = 0.01
            marker.scale.z = 0.01
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.lifetime = rospy.Duration(0.3)
            marker_array.markers.append(marker)
        return marker_array

    def __publish_detected_objects_marker(self, detected_objects_marker: MarkerArray) -> None:
        self.__detected_objects_marker_publisher.publish(detected_objects_marker)