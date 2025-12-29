import rospy
from object_detection.yolo_inference import YoloInference, CentroidEstimator

if __name__ == "__main__":
    try:
        rospy.init_node("yolo_inference_bbox_estimator", anonymous=True)
        yolo_inference = YoloInference(CentroidEstimator.BBOX)
        while not rospy.is_shutdown():
            yolo_inference.run_inference_once()
            rospy.Rate(10).sleep()
    except rospy.ROSInterruptException:
        pass
