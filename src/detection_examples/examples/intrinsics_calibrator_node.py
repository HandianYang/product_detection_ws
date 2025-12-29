from object_detection.intrinsics_calibrator import IntrinsicsCalibrator
import rospy

if __name__ == '__main__':
    rospy.init_node('intrinsics_calibrator_node', anonymous=True)
    intrinsics_calibrator = IntrinsicsCalibrator()
    intrinsics_calibrator.run()