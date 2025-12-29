from object_detection.charuco_detector import CharucoDetector
import rospy

if __name__ == '__main__':
    try:
        rospy.init_node("charuco_detector_node", anonymous=True)
        detector = CharucoDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass