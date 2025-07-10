from object_detection.yolo_inference import YoloInference, CentroidEstimator
import rospy

if __name__ == "__main__":
    try:
        rospy.init_node("yolo_inference_pointcloud_estimator", anonymous=True)
        yolo_inference = YoloInference(CentroidEstimator.POINTCLOUD)
        while not rospy.is_shutdown():
            yolo_inference.get_inference_results()
            rospy.Rate(10).sleep()
    except rospy.ROSInterruptException:
        pass
