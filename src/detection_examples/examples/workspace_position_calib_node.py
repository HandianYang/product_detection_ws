import rospy

from object_detection.workspace_position_calib import WorkspacePositionCalibration

if __name__ == "__main__":
    rospy.init_node("workspace_position_calib_node", anonymous=True)
    obj = WorkspacePositionCalibration()
    rospy.spin()