from pathlib import Path

from geometry_msgs.msg import Pose, TransformStamped
import rospy
import tf
import yaml

class WorkspacePositionCalibration:
    SHELF_YAML = str(Path(__file__).resolve().parents[3]
        / "experiments/config/shelf_aruco_pose.yaml")
    WAREHOUSE_YAML = str(Path(__file__).resolve().parents[3]
        / "experiments/config/warehouse_aruco_pose.yaml")

    def __init__(self):
        self.__shelf_aruco_sub = rospy.Subscriber(
            "/aruco_double/pose", Pose, self.__shelf_aruco_callback)
        self.__warehouse_aruco_sub = rospy.Subscriber(
            "/aruco_double/pose2", Pose, self.__warehouse_aruco_callback)

        self.__tf_listener = tf.TransformListener()
        self.__shelf_last_record_time = None
        self.__warehouse_last_record_time = None

    def __shelf_aruco_callback(self, msg: Pose):
        try:
            (trans, rot) = self.__tf_listener.lookupTransform(
                "tm_base", "shelf_frame", rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return

        data = {
            "translation": {
                "x": trans[0],
                "y": trans[1],
                "z": trans[2]
            },
            "rotation": {
                "x": rot[0],
                "y": rot[1],
                "z": rot[2],
                "w": rot[3]
            }
        }

        if (
            self.__shelf_last_record_time is None
            or rospy.get_rostime() - self.__shelf_last_record_time >= rospy.Duration(1)
        ):
            with open(WorkspacePositionCalibration.SHELF_YAML, "w") as yaml_file:
                yaml.dump(data, yaml_file, default_flow_style=False)
            self.__shelf_last_record_time = rospy.get_rostime()

    def __warehouse_aruco_callback(self, msg: Pose):
        try:
            (trans, rot) = self.__tf_listener.lookupTransform(
                "tm_base", "warehouse_frame", rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return

        data = {
            "translation": {
                "x": trans[0],
                "y": trans[1],
                "z": trans[2]
            },
            "rotation": {
                "x": rot[0],
                "y": rot[1],
                "z": rot[2],
                "w": rot[3]
            }
        }

        if (
            self.__warehouse_last_record_time is None
            or rospy.get_rostime() - self.__warehouse_last_record_time >= rospy.Duration(1)
        ):
            with open(WorkspacePositionCalibration.WAREHOUSE_YAML, "w") as yaml_file:
                yaml.dump(data, yaml_file, default_flow_style=False)
            self.__warehouse_last_record_time = rospy.get_rostime()