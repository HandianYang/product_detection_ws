import rospy
from detection_msgs.msg import BoundingBox, InferenceResult, InferenceResultArray, DetectedObject, DetectedObjectArray
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image, CameraInfo
# from visualization_msgs.msg import Marker, MarkerArray

import cv2
from cv_bridge import CvBridge
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import yaml

import tf2_ros
import tf2_geometry_msgs

from norfair import Detection, Tracker
from scipy.spatial.distance import cdist

class YoloInference():
    DEFAULT_CONFIDENCE_THRESHOLD = 0.7
    DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().parent.parent.parent / "config/class_list.yaml")
    DEFAULT_WEIGHT_PATH = str(Path(__file__).resolve().parent.parent.parent / "weight/yolo11_v1.pt")
    CAMERA_LINK_TO_TIP_LINK_R = np.array([
        [-1,  0,  0],
        [ 0, -1,  0],
        [ 0,  0,  1]
    ])
    CAMERA_LINK_TO_TIP_LINK_T = np.array([0.0175, 0.06, -0.0858])  # TODO: Measure the actual transformation

    def __init__(self):
        self.__confidence_threshold = rospy.get_param("~confidence_threshold", self.DEFAULT_CONFIDENCE_THRESHOLD)
        self.__config_path = rospy.get_param("~config_path", YoloInference.DEFAULT_CONFIG_PATH)
        self.__display_enabled = rospy.get_param("~display_enabled", False)
        self.__weight_path = rospy.get_param("~weight_path", YoloInference.DEFAULT_WEIGHT_PATH)

        self.model = YOLO(self.__weight_path)
        self.class_names = self.__load_class_names_from_yaml()
        self.bridge = CvBridge()
        self.color_image = Image()
        self.depth_image = Image()
        self.camera_intrinsics = CameraInfo()
        self.inference_results = InferenceResultArray()
        self.detected_objects = DetectedObjectArray()

        self.tracker = Tracker(
            distance_function="euclidean",
            distance_threshold=50,
            hit_counter_max=30,
            initialization_delay=0
        )
        self.tracked_objects = []

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.color_image_subscriber = rospy.Subscriber("/camera/color/image_raw", Image, self.__camera_color_image_callback)
        self.depth_image_subscriber = rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.__camera_depth_image_callback)
        self.camera_info_subscriber = rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.__camera_info_callback)

        self.inference_results_publisher = rospy.Publisher("/yolo/inference_results", InferenceResultArray, queue_size=1)
        self.detected_objects_publisher = rospy.Publisher('/yolo/detected_objects', DetectedObjectArray, queue_size=1)
        self.tracked_image_publisher = rospy.Publisher("/yolo/tracked_image", Image, queue_size=1)

    # === Public Methods ===
    def get_inference_results(self) -> None:
        """ Apply YOLO inference and object tracking on the color image and update tracked objects.

        """
        try:
            if not self.color_image.data or self.color_image.height == 0 or self.color_image.width == 0:
                rospy.logwarn("Color image data is empty or has invalid dimensions. Skipping inference.")
                return
            bgr_image = np.frombuffer(self.color_image.data, dtype=np.uint8).reshape(self.color_image.height, self.color_image.width, -1)
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            predictions = self.model.predict(
                source=rgb_image,
                conf=self.__confidence_threshold,
                show=self.__display_enabled,
                save=False,
                verbose=False)[0]
        except ValueError as e:
            rospy.logerr(f"Error processing color image: {e}")
            return

        #  Create InferenceResultArray with predictions and update tracker
        self.inference_results = InferenceResultArray()
        tracker_detections = []
        for prediction in predictions.boxes:
            inference_result = InferenceResult()
            inference_result.label = self.class_names[int(prediction.cls[0].item())]
            inference_result.confidence = prediction.conf[0].item()
            inference_result.bbox = self.get_bounding_box_from_prediction(prediction)
            self.inference_results.results.append(inference_result)
            tracker_detections.append(
                Detection(
                    points=np.array([inference_result.bbox.center_x, inference_result.bbox.center_y]),
                    scores=np.array([inference_result.confidence])
                )
            )
        self.tracked_objects = self.tracker.update(tracker_detections)

        # Assign Norfair IDs to inference results based on tracked objects
        track_centers = np.array([obj.estimate[0] for obj in self.tracked_objects])
        pred_centers = np.array([
            [res.bbox.center_x, res.bbox.center_y]
            for res in self.inference_results.results
        ])
        if len(track_centers) > 0 and len(pred_centers) > 0:
            distances = cdist(track_centers, pred_centers)
            matched_preds = set()
            matched_tracks = set()

            for track_idx, pred_idx in zip(*np.where(distances < 20)):
                if track_idx in matched_tracks or pred_idx in matched_preds:
                    continue

                self.inference_results.results[pred_idx].id = self.tracked_objects[track_idx].id

                matched_tracks.add(track_idx)
                matched_preds.add(pred_idx)
        self.inference_results_publisher.publish(self.inference_results)

    def transform_bbox_to_position(self) -> None:
        if not self.inference_results.results:
            return
        
        self.detected_objects = DetectedObjectArray()
        for result in self.inference_results.results:
            detected_object = DetectedObject()
            detected_object.id = result.id
            detected_object.label = result.label
            detected_object.confidence = result.confidence
            detected_object.position = self.__get_position_wrt_base_link(result.bbox)
            if detected_object.position is None:
                rospy.logwarn(f"Skipping object {result.label} due to invalid position.")
                continue
            self.detected_objects.objects.append(detected_object)
        self.detected_objects_publisher.publish(self.detected_objects)

    def publish_image_with_tracked_objects_marker(self) -> None:
        if not self.color_image.data or self.color_image.height == 0 or self.color_image.width == 0:
            return

        cv_image = self.bridge.imgmsg_to_cv2(self.color_image, desired_encoding="bgr8")
        for obj in self.tracked_objects:
            x, y = obj.estimate[0]
            cv2.circle(cv_image, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(cv_image, f'ID: {obj.id}', (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        image_msg_with_marker = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        self.tracked_image_publisher.publish(image_msg_with_marker)

    # === Private Methods (initialization) ===
    def __load_class_names_from_yaml(self) -> list:
        with open(self.__config_path, 'r') as f:
            data = yaml.safe_load(f)
        return data['class_names']

    # === Private Methods (supports transform_bbox_to_position()) ===
    def __get_position_wrt_base_link(self, bbox: BoundingBox) -> Point:
        position_camera_link = self.__get_position_wrt_camera_link(bbox)
        if position_camera_link is None:
            # rospy.logerr("Failed to compute position with respect to camera link.")
            return None
        position_tip_link = self.__transfrom_camera_link_to_tip_link(position_camera_link)
        position_base_link = self.__transform_tip_link_to_base_link(position_tip_link)
        if position_base_link is None:
            # rospy.logerr("Failed to transform position to base link.")
            return None
        return position_base_link

    def __get_position_wrt_camera_link(self, bbox: BoundingBox) -> Point:
        u = int(bbox.center_x)
        v = int(bbox.center_y)
        depth = self.__compute_depth_from_bbox(bbox)
        if depth is None:
            rospy.logerr("No valid depth found for bbox.")
            return None

        fx = self.camera_intrinsics['fx']
        fy = self.camera_intrinsics['fy']
        cx = self.camera_intrinsics['cx']
        cy = self.camera_intrinsics['cy']
        
        position = Point()
        position.x = (u - cx) * depth / fx
        position.y = (v - cy) * depth / fy
        position.z = depth
        return position

    def __compute_depth_from_bbox(self, bbox: BoundingBox) -> float:
        u_min = int(bbox.center_x) - int(bbox.width / 2)
        u_max = int(bbox.center_x) + int(bbox.width / 2)
        v_min = int(bbox.center_y) - int(bbox.height / 2)
        v_max = int(bbox.center_y) + int(bbox.height / 2)
        patch = self.depth_image[v_min:v_max, u_min:u_max]
        valid = patch[(patch > 0.2) & (patch < 2)]

        if len(valid) < 5:
            return None
        else:
            return float(np.median(valid))

    def __transfrom_camera_link_to_tip_link(self, position_camera_link: Point) -> Point:
        position_camera_link_np = np.array([position_camera_link.x, position_camera_link.y, position_camera_link.z])
        position_tip_link_np = YoloInference.CAMERA_LINK_TO_TIP_LINK_R @ position_camera_link_np \
            + YoloInference.CAMERA_LINK_TO_TIP_LINK_T
        
        position_tip_link = Point()
        position_tip_link.x = position_tip_link_np[0]
        position_tip_link.y = position_tip_link_np[1]
        position_tip_link.z = position_tip_link_np[2]
        return position_tip_link

    def __transform_tip_link_to_base_link(self, position_tip_link: Point) -> Point:
        try:
            if not self.tf_buffer.can_transform('base_link', 'tm_tip_link', rospy.Time(0), rospy.Duration(1.0)):
                rospy.logwarn("Cannot transform from tm_tip_link to base_link")
                return None
            point_stamped = tf2_geometry_msgs.PointStamped()
            point_stamped.header.frame_id = 'tm_tip_link'
            point_stamped.header.stamp = rospy.Time(0)
            point_stamped.point = position_tip_link
            transformed_point_stamped = self.tf_buffer.transform(point_stamped, 'base_link', rospy.Duration(1.0))
            return transformed_point_stamped.point
        except Exception as e:
            rospy.logerr(f"TF2 transform error: {e}")
            return None

    # === Private Methods (callbacks) ===
    def __camera_color_image_callback(self, color_image: Image) -> None:
        if not color_image.data:
            rospy.logwarn("No data received for color image. Skipping this frame.")
            return
        self.color_image = color_image
    
    def __camera_depth_image_callback(self, depth_image: Image) -> None:
        try:
            # Convert ROS image to OpenCV (16UC1 or 32FC1 depending on camera)
            self.depth_image = self.bridge.imgmsg_to_cv2(depth_image, desired_encoding="passthrough")
        except Exception as e:
            rospy.logerr(f"Depth image conversion error: {e}")
            return

        # Convert to meters if needed
        if self.depth_image.dtype == np.uint16:
            self.depth_image = self.depth_image.astype(np.float32) / 1000.0

    def __camera_info_callback(self, msg: CameraInfo) -> None:
        self.camera_intrinsics = {
            'fx': msg.K[0],
            'fy': msg.K[4],
            'cx': msg.K[2],
            'cy': msg.K[5]
        }

    @staticmethod
    def get_bounding_box_from_prediction(result) -> BoundingBox:
        bbox = BoundingBox()
        x1, y1, x2, y2 = result.xyxy[0].tolist()
        bbox.center_x = (x1 + x2) / 2.0
        bbox.center_y = (y1 + y2) / 2.0
        bbox.width = x2 - x1
        bbox.height = y2 - y1
        return bbox