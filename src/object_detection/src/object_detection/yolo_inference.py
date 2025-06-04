import rospy
from detection_msgs.msg import BoundingBox, BoundingBoxArray
from sensor_msgs.msg import Image

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import yaml

class YoloInference():
    def __init__(self):
        self.DEFAULT_CONFIDENCE_THRESHOLD = 0.7
        self.DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().parent.parent.parent / "config/class_list.yaml")
        self.DEFAULT_WEIGHT_PATH = str(Path(__file__).resolve().parent.parent.parent / "weight/yolo11_v1.pt")
        self.parse_parameters()

        self.model = YOLO(self.weight_path)
        self.class_names = self.load_class_names()
        self.color_image = Image()

        self.boundingbox_publisher = rospy.Publisher("/yolo/boundingbox", BoundingBoxArray, queue_size=10)
        self.color_image_subscriber = rospy.Subscriber("/camera/color/image_raw", Image, self.camera_color_image_callback)

    def parse_parameters(self) -> None:
        self.config_path = rospy.get_param("~config_path", self.DEFAULT_CONFIG_PATH)
        self.weight_path = rospy.get_param("~weight_path", self.DEFAULT_WEIGHT_PATH)
        self.confidence_threshold = rospy.get_param("~confidence_threshold", self.DEFAULT_CONFIDENCE_THRESHOLD)
        self.display_enabled = rospy.get_param("~display_enabled", True)
        
    def load_class_names(self) -> list:
        with open(self.config_path, 'r') as f:
            data = yaml.safe_load(f)
        return data['class_names']

    def camera_color_image_callback(self, color_image: Image) -> None:
        self.color_image = color_image

    def get_inference_results(self) -> None:
        if not self.color_image.data:
            rospy.logwarn("No data received for color image. Skipping this frame.")
            return

        try:
            bgr_image = np.frombuffer(self.color_image.data, dtype=np.uint8).reshape(self.color_image.height, self.color_image.width, -1)
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            results = self.model.predict(
                source=rgb_image,
                conf=self.confidence_threshold,
                show=self.display_enabled,
                save=False,
                verbose=False)[0]
        except ValueError as e:
            rospy.logerr(f"Error processing color image: {e}")
            return

        boundingboxes = BoundingBoxArray()
        for result in results.boxes:
            boundingbox = BoundingBox()
            boundingbox.label = self.class_names[int(result.cls[0].item())]
            boundingbox.confidence = result.conf[0].item()
            boundingbox.bbox = self.get_bounding_box_boundary(result)
            boundingboxes.bboxes.append(boundingbox)
        self.boundingbox_publisher.publish(boundingboxes)

    def get_bounding_box_boundary(self, result) -> list:
        x1, y1, x2, y2 = result.xyxy[0].tolist()
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
        return [x_center, y_center, width, height]