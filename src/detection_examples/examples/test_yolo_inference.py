import rospy
from object_detection.yolo_inference import YoloInference

if __name__ == "__main__":
    try:
        rospy.init_node("test_yolo_inference", anonymous=True)
        yolo_inference = YoloInference()
        while not rospy.is_shutdown():
            yolo_inference.get_inference_results()
            yolo_inference.transform_bbox_to_position()
            yolo_inference.publish_image_with_tracked_objects_marker()
            rospy.Rate(20).sleep()
    except rospy.ROSInterruptException:
        pass
