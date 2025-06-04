from object_detection.yolo_inference import YoloInference
import rospy

if __name__ == "__main__":
    try:
        rospy.init_node("test_yolo_inference", anonymous=True)
        yolo_inference = YoloInference()
        while not rospy.is_shutdown():
            yolo_inference.get_inference_results()
            rospy.Rate(30).sleep()
    except rospy.ROSInterruptException:
        pass
