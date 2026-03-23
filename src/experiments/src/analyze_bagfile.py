from abc import ABC, abstractmethod
import csv
import os
from pathlib import Path
import sys
import yaml

from geometry_msgs.msg import Point
import rosbag

USAGE_PROMPT_MESSAGE = "\
  Usage: python3 analyze_bagfile.py <user>_<ctrl><task><trial>[_rX]\n\
    <user>: 01, 02, ..., 99\n\
    <ctrl>: m (manual control), s (shared control)\n\
    <task>: s (single item), m (multiple items), r (replenishment task)\n\
    <trial>: 01, 02, ..., 99\n\n\
  Example: \n\
    python3 analyze_bagfile.py 01_ms01 (manual-single)\n\
    python3 analyze_bagfile.py 02_sm03 (shared-multiple)\n"

CONTROL_TYPE_NAME = {
    'm': "manual",
    's': "shared"
}

TASK_TYPE_NAME = {
    's': "single",
    'm': "multiple",
    'r': "replenishment"
}

TRIGGER_THRESHOLD = 0.8
HEIGHT_TOLERANCE = 0.15     # [m]
POSITION_TOLERANCE = 0.05  # [m]
PICK_AREA_RADIUS = 0.3  # [m]
LIFT_DISTANCE = 0.05    # [m]

GRIPPER_TOPIC_NAME = '/gripper/cmd_gripper'
GRIPPER_POSE_TOPIC_NAME = '/marslite_control/gripper_pose'
JOY_TOPIC_NAME = '/vr_controller/joy'
OBJECTS_TOPIC_NAME = '/marslite_control/objects_with_belief'
RECORD_TOPIC_NAME = '/marslite_control/record_signal'
RESTART_TOPIC_NAME = '/marslite_control/restart_attempt_signal'

SHELF_ARUCO_POSE_YAML = str(Path(__file__).resolve().parents[1]
    / "config/shelf_aruco_pose.yaml")
WAREHOUSE_ARUCO_POSE_YAML = str(Path(__file__).resolve().parents[1]
    / "config/warehouse_aruco_pose.yaml")

class Zone:
    def __init__(self, x_min: float = 0.0, x_max: float = 0.0, y_min: float = 0.0, y_max: float = 0.0):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

class TaskAnalyzer(ABC):
    def __init__(self, bagfile_path: str):
        # experiment data
        self._data = dict()
        self._data["Subject_ID"] = str(int(bagfile_path[-11:-9]))
        self._data["Control"] = CONTROL_TYPE_NAME[bagfile_path[-8]]
        self._data["Task"] = TASK_TYPE_NAME[bagfile_path[-7]]
        self._data["Trial"] = str(int(bagfile_path[-6:-4]))

        # input/output files
        self._bag = rosbag.Bag(bagfile_path)
        self._csv_folder = str(Path(__file__).resolve().parent.parent) + "/data"
        self._csv_path = self._csv_folder + "/" + bagfile_path[-12:-9] \
            + "_" + self._data["Task"] + ".csv"
        self._headers = list()

        # workspace setup
        with open(SHELF_ARUCO_POSE_YAML, 'r') as f:
            data = yaml.safe_load(f)
        self._shelf_aruco_position = data["translation"]
        with open(WAREHOUSE_ARUCO_POSE_YAML, 'r') as f:
            data = yaml.safe_load(f)
        self._warehouse_aruco_position = data["translation"]
        self._pick_zone = Zone()
        self._place_zone = Zone()
        
        # analyzer variables
        self._timeline = dict()
        self._gripper_position = Point()
        self._is_gripper_closed = False
        self._objects_status = dict()  # {id: {label, centroid, picked, placed}}
        self._topic_list = [
            GRIPPER_TOPIC_NAME,
            GRIPPER_POSE_TOPIC_NAME,
            JOY_TOPIC_NAME,
            OBJECTS_TOPIC_NAME,
            RECORD_TOPIC_NAME,
            RESTART_TOPIC_NAME
        ]
    
    @staticmethod
    def get_distance(pos1: Point, pos2: Point) -> float:
        dx = pos1.x - pos2.x
        dy = pos1.y - pos2.y
        dz = pos1.z - pos2.z
        return (dx * dx + dy * dy + dz * dz) ** 0.5
    
    @staticmethod
    def get_xy_distance(pos1: Point, pos2: Point) -> float:
        dx = pos1.x - pos2.x
        dy = pos1.y - pos2.y
        return (dx * dx + dy * dy) ** 0.5
    
    @staticmethod
    def get_z_distance(pos1: Point, pos2: Point) -> float:
        return abs(pos1.z - pos2.z)

    @abstractmethod
    def analyze(self):
        pass

    def print_trial_data(self):
        '''
        Print data of one trial.
        '''
        for key, value in self._data.items():
            print(f"{key}:\t{value}")
    
    def export_trial_to_csv(self):
        '''
        Append data of one trial to the specifc CSV file.
        '''
        row_data = [self._data.get(key, "") for key in self._headers]        
        with open(self._csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)
        print(f"Export data to the CSV file {self._csv_path}.")

    def _create_csv_file(self):
        '''
        NOTE: `self._csv_folder` and `self._headers` should be defined beforehand
        '''
        os.makedirs(self._csv_folder, exist_ok=True)
        if not os.path.exists(self._csv_path):
            with open(self._csv_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self._headers)
            print(f"Create new CSV file {self._csv_path}.")

    def _is_in_pick_area(self) -> bool:
        '''
        Detect whether the gripper is inside the pick area.

        The definition of the pick area differs according to the task type.
        Please refer to the child classes to check area definition.
        '''
        # for _, obj_info in self._objects_status.items():
        #     obj_dist = self.get_distance(self._gripper_position, obj_info['centroid'])
        #     if obj_dist < PICK_AREA_RADIUS:
        #         return True
        # return False
    
        return self._gripper_position.x >= self._pick_zone.x_min \
            and self._gripper_position.x <= self._pick_zone.x_max \
            and self._gripper_position.y >= self._pick_zone.y_min \
            and self._gripper_position.y <= self._pick_zone.y_max 

    def _is_in_place_area(self) -> bool:
        '''
        Detect whether the gripper is inside the place area.

        The definition of the place area differs according to the task type.
        Please refer to the child classes to check area definition.
        '''
        return self._gripper_position.x >= self._place_zone.x_min \
            and self._gripper_position.x <= self._place_zone.x_max \
            and self._gripper_position.y >= self._place_zone.y_min \
            and self._gripper_position.y <= self._place_zone.y_max 

    def _is_object_reached(self, object_position: Point) -> bool:
        '''
        Detect whether the gripper is close to the object. 

        Looser tolerance is applied to the Z position difference so that user
        can decide the grasping height.
        '''
        xy_dist = self.get_xy_distance(self._gripper_position, object_position)
        z_dist = self.get_z_distance(self._gripper_position, object_position)
        # print(f"[DEBUG] xy_dist: {xy_dist}")
        # print(f"[DEBUG] z_dist: {z_dist}")
        return xy_dist < POSITION_TOLERANCE and z_dist < HEIGHT_TOLERANCE

    @abstractmethod
    def _is_task_completed(self):
        pass

    @abstractmethod
    def _calculate_operation_time(self):
        pass


class SingleObjectGraspingAnalyzer(TaskAnalyzer):
    PICK_ZONE_LENGTH = 0.6
    PICK_ZONE_WIDTH = 0.4
    PICK_ZONE_OFFSET = 0.3  # width offset area extending towards the user

    def __init__(self, bagfile_path: str):
        super().__init__(bagfile_path)

        # Define pick zone for single object grasping tasks
        self._pick_zone.x_min = self._shelf_aruco_position["x"] - self.PICK_ZONE_LENGTH / 2
        self._pick_zone.x_max = self._shelf_aruco_position["x"] + self.PICK_ZONE_LENGTH / 2
        self._pick_zone.y_min = self._shelf_aruco_position["y"] - self.PICK_ZONE_WIDTH - self.PICK_ZONE_OFFSET
        self._pick_zone.y_max = self._shelf_aruco_position["y"]
        # NOTE: No definitaion for place zone in single object grasping tasks!

        # Define headers for single object grasping tasks
        self._headers = [
            "Subject_ID", "Control", "Task", "Trial",   # fixed in trial
            "Attempt", "Object_Type", "Is_Success",
            "Pick_Time",    # A
            "Lift_Time",    # a
            "Total_Time",   # A + a
            "Error_Msg"
        ]
        self._create_csv_file()

        # Define critical timestamps
        self._timeline["start"] = None      # gripper in pick area
        self._timeline["picked"] = None     # object is picked
        self._timeline["lifted"] = None     # object is lifted

    def analyze(self) -> None:
        '''
        Analyze and obtain data from one trial of single object grasping tasks.
        '''
        is_ready_to_record_object_position = False
        is_ready_to_start_timer = False
        # last_timer_start_time = None

        for topic, msg, time in self._bag.read_messages(topics=self._topic_list):    
            if topic == GRIPPER_TOPIC_NAME:
                self._is_gripper_closed = msg.data
                if self._is_gripper_closed:
                    print("[DEBUG] Gripper is closed")
            
            elif topic == RECORD_TOPIC_NAME:
                is_ready_to_record_object_position = True

            elif topic == OBJECTS_TOPIC_NAME:
                if msg.objects and is_ready_to_record_object_position:
                    self._objects_status.clear()
                    self._objects_status[0] = {
                        "label": msg.objects[0].label,  # string
                        "centroid": msg.objects[0].centroid,  # geoemtry_msgs.Point
                        "picked": False
                    }
                    self._data["Object_Type"] = msg.objects[0].label
                    is_ready_to_record_object_position = False
            
            elif topic == JOY_TOPIC_NAME:
                # Start the timer at the beginning of the attempt, and
                #  after the user starts to move the robot
                if msg.axes[2] > TRIGGER_THRESHOLD: # primary index trigger
                    if is_ready_to_start_timer:
                        print("[DEBUG] Start the timer")
                        self._timeline["start"] = time.to_sec()
                        is_ready_to_start_timer = False
            
            elif topic == RESTART_TOPIC_NAME:
                if "Attempt" not in self._data:
                    # The button is pressed first time
                    self._data["Attempt"] = 1
                else:
                    self._calculate_operation_time()
                    self.export_trial_to_csv()
                    self._data["Attempt"] += 1

                self._data["Is_Success"] = 0
                self._data["Pick_Time"] = "nan"
                self._data["Lift_Time"] = "nan"
                self._data["Total_Time"] = "nan"
                self._data["Error_Msg"] = ""
                is_ready_to_start_timer = True
            
            elif topic == GRIPPER_POSE_TOPIC_NAME:
                self._gripper_position = msg.pose.position
                # if is_ready_to_start_timer and self._is_in_pick_area():
                #     # Start the timer
                #     self._timeline["start"] = time.to_sec()
                #     is_ready_to_start_timer = False
                if (
                    self._is_gripper_closed
                    and self._timeline["start"] is not None
                    and self._is_object_reached(self._objects_status[0]["centroid"])
                    and not self._objects_status[0]["picked"]
                ):
                    # Mark the object as picked
                    print("[DEBUG] Mark the object as picked")
                    self._timeline["picked"] = time.to_sec()
                    self._objects_status[0]["picked"] = True

            if self._is_task_completed():
                self._timeline["lifted"] = time.to_sec()
                self._data["Is_Success"] = 1
                break
        
        self._bag.close()
        self._calculate_operation_time()
        self.print_trial_data()
        self.export_trial_to_csv()
        
    def _is_task_completed(self) -> bool:
        '''
        Decide whether the grasping task is completed.

        The conditions for single object grasping tasks are:
        1. The timer has started
        2. The object has been grasped
        3. The object has been lifted up for enough distance
        '''
        return self._timeline["start"] is not None \
            and self._objects_status \
            and self._objects_status[0]["picked"] \
            and (self._gripper_position.z - self._objects_status[0]["centroid"].z) >= LIFT_DISTANCE

    def _calculate_operation_time(self) -> None:
        '''
        Calculate the overall operation time for this trial for single object
        grasping tasks.

        - Pick time := t2 - t1
        - Lift time := t3 - t2
        - Total time := t3 - t1
        '''
        t1 = self._timeline["start"]
        t2 = self._timeline["picked"]
        t3 = self._timeline["lifted"]
        if t1 is not None and t2 is not None:
            self._data["Pick_Time"] = t2 - t1
            if t3 is not None:
                self._data["Lift_Time"] = t3 - t2
                self._data["Total_Time"] = t3 - t1


    
class MultipleObjectGraspingAnalyzer(TaskAnalyzer):
    PICK_ZONE_LENGTH = 0.6
    PICK_ZONE_WIDTH = 0.3
    PICK_ZONE_OFFSET = 0.2  # width offset area extending towards the user
    PLACE_ZONE_LENGTH = 0.2
    PLACE_ZONE_OFFSET = 0.08    # surrounding area

    def __init__(self, bagfile_path: str):
        super().__init__(bagfile_path)

        # Define place zone
        self._pick_zone.x_min = self._shelf_aruco_position["x"] - self.PICK_ZONE_LENGTH / 2
        self._pick_zone.x_max = self._shelf_aruco_position["x"] + self.PICK_ZONE_LENGTH / 2
        self._pick_zone.y_min = self._shelf_aruco_position["y"] - self.PICK_ZONE_OFFSET
        self._pick_zone.y_max = self._shelf_aruco_position["y"] + self.PICK_ZONE_WIDTH
        # Define place zone
        self._place_zone.x_min = self._warehouse_aruco_position["x"] - self.PLACE_ZONE_OFFSET
        self._place_zone.x_max = self._warehouse_aruco_position["x"] + self.PLACE_ZONE_LENGTH + self.PLACE_ZONE_OFFSET
        self._place_zone.y_min = self._warehouse_aruco_position["y"] - self.PLACE_ZONE_LENGTH / 2 - self.PLACE_ZONE_OFFSET
        self._place_zone.y_max = self._warehouse_aruco_position["y"] + self.PLACE_ZONE_LENGTH / 2 + self.PLACE_ZONE_OFFSET

        # Define headers for multiple object grasping tasks
        self._headers = [
            "Subject_ID", "Control", "Task", "Trial",   # fixed in trial
            "Sequence", "Attempt", "Object_Type", "Is_Success", 
            "Pick_Time",    # A
            "Place_Time",   # B
            "Return_Time",  # C
            "Total_Time",   # A + B + C
            "Total_Efficient_Time", # A + B
            "Error_Msg"
        ]
        self._create_csv_file()

        # Define critical timestamps
        self._timeline["start"] = None
        self._timeline["picked"] = None
        self._timeline["placed"] = None
        self._timeline["returned"] = None
    
    def analyze(self) -> None:
        target_id = -1
        is_ready_to_record_object_position = False
        is_ready_to_start_timer = False
        last_timer_start_time = None

        for topic, msg, time in self._bag.read_messages(topics=self._topic_list):
            if topic == GRIPPER_TOPIC_NAME:
                self._is_gripper_closed = msg.data
            
            elif topic == OBJECTS_TOPIC_NAME:
                if is_ready_to_record_object_position and msg.objects:
                    self._objects_status.clear()
                    for index, obj in enumerate(msg.objects):
                        self._objects_status[index] = {
                            "label": obj.label, # string
                            "centroid": obj.centroid,   # geometry_msgs.Point
                            "picked": False,
                            "placed": False
                        }
                    is_ready_to_record_object_position = False

            elif topic == JOY_TOPIC_NAME:
                # [NOTE] The user can restart this trial by pressing the stick
                #   button again
                if msg.buttons[2] == 1:     # analog stick button
                    if "Attempt" not in self._data:
                        self._data["Sequence"] = 1
                        self._data["Attempt"] = 1                        
                    elif (
                        last_timer_start_time is not None
                        and time.to_sec() - last_timer_start_time >= 3
                    ):
                        self._calculate_operation_time()
                        self.export_trial_to_csv()
                        self._data["Attempt"] += 1
                    
                    self._data["Is_Success"] = 0
                    self._data["Pick_Time"] = "nan"
                    self._data["Place_Time"] = "nan"
                    self._data["Return_Time"] = "nan"
                    self._data["Total_Time"] = "nan"
                    self._data["Total_Efficient_Time"] = "nan"
                    self._data["Error_Msg"] = ""
                    is_ready_to_record_object_position = True
                    is_ready_to_start_timer = True
                    last_timer_start_time = time.to_sec()

            elif topic == GRIPPER_POSE_TOPIC_NAME:
                self._gripper_position = msg.pose.position

                ### Phase 1: Approach to pick area
                if is_ready_to_start_timer and self._is_in_pick_area():
                    if self._timeline["placed"] is None:
                        # the beginning of the first object
                        self._timeline["start"] = time.to_sec()
                    else:
                        # the end of this object, and the beginning of the next
                        #  object
                        self._timeline["returned"] = time.to_sec()
                        self._data["Is_Success"] = 1
                        self._calculate_operation_time()
                        self.export_trial_to_csv()
                        self._timeline["start"] = self._timeline["returned"]

                        # Reset this trial
                        self._data["Sequence"] = self._data["Sequence"] + 1
                        self._data["Attempt"] = 1
                        self._data["Is_Success"] = 0
                        self._data["Pick_Time"] = "nan"
                        self._data["Place_Time"] = "nan"
                        self._data["Return_Time"] = "nan"
                        self._data["Total_Time"] = "nan"
                        self._data["Total_Efficient_Time"] = "nan"
                        self._data["Error_Msg"] = ""
                    is_ready_to_start_timer = False

                ### Phase 2: Approach to the object
                if (
                    target_id == -1
                    and self._timeline["start"] is not None
                    and (
                        self._timeline["picked"] is None
                        or self._timeline["picked"] < self._timeline["start"]
                    )
                    and self._is_gripper_closed
                ):
                    # Find candidates of the picked object
                    candidates = {} # id : xy_dist
                    for obj_id, obj_info in self._objects_status.items():
                        if (
                            not obj_info["picked"]
                            and self._is_object_reached(obj_info["centroid"])
                        ):
                            candidates[obj_id] = self.get_xy_distance(
                                self._gripper_position,
                                obj_info["centroid"]
                            )

                    if not candidates:
                        continue
                    target_id = min(candidates, key=candidates.get)
                    self._objects_status[target_id]["picked"] = True
                    self._timeline["picked"] = time.to_sec()
                    self._data["Object_Type"] = self._objects_status[target_id]["label"]

                ### Phase 3: Transfer to the place area
                if target_id != -1 and not self._is_gripper_closed:
                    if self._is_in_place_area():
                        self._objects_status[target_id]["placed"] = True
                        self._timeline["placed"] = time.to_sec()
                        target_id = -1
                        is_ready_to_start_timer = True
                    else:
                        # NOTE: reset this attempt
                        self._data["Error_Msg"] = "Object misplaced"
            
            if self._is_task_completed():
                self._data["Is_Success"] = 1
                break
        
        self._bag.close()
        self._calculate_operation_time()
        self.export_trial_to_csv()
    
    def _is_task_completed(self) -> bool:
        '''
        Decide whether the grasping task is completed.

        The conditions for multiple object grasping tasks are:
        1. The timer has started
        2. The object list is not empty
        3. All objects have been picked and placed
        '''
        return self._timeline["start"] is not None \
            and self._objects_status \
            and all(obj_info['picked'] for obj_info in self._objects_status.values()) \
            and all(obj_info['placed'] for obj_info in self._objects_status.values())


    def _calculate_operation_time(self) -> None:
        '''
        Calculate the overall operation time for this trial for multiple object
        grasping tasks.

        - Pick time := t2 - t1
        - Place time := t3 - t2
        - Return time := t4 - t3
        '''
        t1 = self._timeline["start"]
        t2 = self._timeline["picked"]
        t3 = self._timeline["placed"]
        t4 = self._timeline["returned"]
        if t1 is not None and t2 is not None and t1 < t2:
            self._data["Pick_Time"] = t2 - t1
            if t3 is not None and t2 < t3:
                self._data["Place_Time"] = t3 - t2
                self._data["Total_Efficient_Time"] = t3 - t1
                if t4 is not None and t3 < t4:
                    self._data["Return_Time"] = t4 - t3
                    self._data["Total_Time"] = t4 - t1


class ReplenishmentTaskAnalyzer(TaskAnalyzer):
    PICK_ZONE_LENGTH = 0.36
    PICK_ZONE_WIDTH = 0.25
    PICK_ZONE_OFFSET = 0.1  # length offset area extending towards the robot
    PLACE_ZONE_LENGTH = 0.63
    PLACE_ZONE_WIDTH = 0.38
    PLACE_ZONE_OFFSET = 0.05  # width offset area extending towards the user

    def __init__(self, bagfile_path: str):
        super().__init__(bagfile_path)
        self._new_id_counter = 0    # id counter for new objects

        # Define pick zone
        self._pick_zone.x_min=self._shelf_aruco_position.x - self.PICK_ZONE_LENGTH - self.PICK_ZONE_OFFSET
        self.x_max=self._shelf_aruco_position.x
        self.y_min=self._shelf_aruco_position.y - self.PICK_ZONE_WIDTH / 2
        self.y_max=self._shelf_aruco_position.y + self.PICK_ZONE_WIDTH / 2
        # Define place zone
        self._place_zone.x_min=self._warehouse_aruco_position.x - self.PLACE_ZONE_LENGTH / 2
        self._place_zone.x_max=self._warehouse_aruco_position.x + self.PLACE_ZONE_LENGTH / 2
        self._place_zone.y_min=self._warehouse_aruco_position.y - self.PLACE_ZONE_OFFSET,
        self._place_zone.y_max=self._warehouse_aruco_position.y + self.PLACE_ZONE_WIDTH

        # Define headers for multiple object grasping tasks
        self._headers = [
            "Subject_ID", "Control", "Task", "Trial",   # fixed in trial
            "Sequence", "Attempt", "Object_Type", "Is_Success", 
            "Pick_Time",    # A
            "Place_Time",   # B
            "Return_Time",  # C
            "Total_Time",   # A + B + C
            "Total_Efficient_Time", # A + B
            "Error_Msg"
        ]
        self._create_csv_file()

        # Define critical timestamps
        self._timeline["start"] = None
        self._timeline["picked"] = None
        self._timeline["placed"] = None
        self._timeline["returned"] = None


    
    def analyze(self) -> None:
        # TODO: Fix the occlusion problem
        target_id = -1
        is_ready_to_record_object_position = False
        is_ready_to_start_timer = False
        last_timer_start_time = None

        for topic, msg, time in self._bag.read_messages(topics=self._topic_list):
            if topic == GRIPPER_TOPIC_NAME:
                self._is_gripper_closed = msg.data
            
            elif topic == OBJECTS_TOPIC_NAME:
                if is_ready_to_record_object_position and msg.objects:
                    for new_obj in msg.objects:
                        # typeof(new_obj) == detection_msgs.DetectedObject
                        is_new_object = True
                        for old_id, old_obj in self._objects_status.items():
                            # typeof(old_obj) == dict()
                            if new_obj.label != old_obj["label"]:
                                continue
                            dx = new_obj.centroid.x - old_obj["centroid"].x
                            dy = new_obj.centroid.y - old_obj["centroid"].y
                            dz = new_obj.centroid.z - old_obj["centroid"].z
                            dist = (dx**2 + dy**2 + dz**2) ** 0.5
                            if dist < 0.03:
                                is_new_object = False
                                old_obj["centroid"] = new_obj.centroid
                                break
                        
                        if is_new_object:
                            self._objects_status[self._new_id_counter] = {
                                "label": new_obj.label,
                                "centroid": new_obj.centroid,
                                "picked": False,
                                "placed": False
                            }
                            self._new_id_counter += 1
                    is_ready_to_record_object_position = False

            elif topic == JOY_TOPIC_NAME:
                # [NOTE] The user can restart this trial by pressing the stick
                #   button again
                if msg.buttons[2] == 1:     # analog stick button
                    if "Attempt" not in self._data:
                        self._data["Sequence"] = 1
                        self._data["Attempt"] = 1                        
                    elif (
                        last_timer_start_time is not None
                        and time.to_sec() - last_timer_start_time >= 3
                    ):
                        self._calculate_operation_time()
                        self.export_trial_to_csv()
                        self._data["Attempt"] += 1
                    
                    self._data["Is_Success"] = 0
                    self._data["Pick_Time"] = "nan"
                    self._data["Place_Time"] = "nan"
                    self._data["Return_Time"] = "nan"
                    self._data["Total_Time"] = "nan"
                    self._data["Total_Efficient_Time"] = "nan"
                    self._data["Error_Msg"] = ""
                    is_ready_to_record_object_position = True
                    is_ready_to_start_timer = True
                    last_timer_start_time = time.to_sec()

            elif topic == GRIPPER_POSE_TOPIC_NAME:
                self._gripper_position = msg.pose.position

                ### Phase 1: Approach to pick area
                if is_ready_to_start_timer and self._is_in_pick_area():
                    if self._timeline["placed"] is None:
                        # the beginning of the first object
                        self._timeline["start"] = time.to_sec()
                    else:
                        # the end of this object, and the beginning of the next
                        #  object
                        self._timeline["returned"] = time.to_sec()
                        self._data["Is_Success"] = 1
                        self._calculate_operation_time()
                        self.export_trial_to_csv()
                        self._timeline["start"] = self._timeline["returned"]

                        # Reset this trial
                        self._data["Sequence"] = self._data["Sequence"] + 1
                        self._data["Attempt"] = 1
                        self._data["Is_Success"] = 0
                        self._data["Pick_Time"] = "nan"
                        self._data["Place_Time"] = "nan"
                        self._data["Return_Time"] = "nan"
                        self._data["Total_Time"] = "nan"
                        self._data["Total_Efficient_Time"] = "nan"
                        self._data["Error_Msg"] = ""
                    is_ready_to_start_timer = False

                ### Phase 2: Approach to the object
                if (
                    target_id == -1
                    and self._timeline["start"] is not None
                    and (
                        self._timeline["picked"] is None
                        or self._timeline["picked"] < self._timeline["start"]
                    )
                    and self._is_gripper_closed
                ):
                    # Find canadidates of the picked object
                    candidates = {} # id : xy_dist
                    for obj_id, obj_info in self._objects_status.items():
                        if (
                            not obj_info["picked"]
                            and self._is_object_reached(obj_info["centroid"])
                        ):
                            candidates[obj_id] = self.get_xy_distance(
                                self._gripper_position,
                                obj_info["centroid"]
                            )

                    if not candidates:
                        continue
                    target_id = min(candidates, key=candidates.get)
                    self._objects_status[target_id]["picked"] = True
                    self._timeline["picked"] = time.to_sec()
                    self._data["Object_Type"] = self._objects_status[target_id]["label"]

                ### Phase 3: Transfer to the place area
                if target_id != -1 and not self._is_gripper_closed:
                    if self._is_in_place_area():
                        self._objects_status[target_id]["placed"] = True
                        self._timeline["placed"] = time.to_sec()
                        target_id = -1
                        is_ready_to_start_timer = True
                    else:
                        self._data["Error_Msg"] = "Object misplaced"
            
            if self._is_task_completed():
                self._data["Is_Success"] = 1
                break
        
        self._bag.close()
        self._calculate_operation_time()
        self.export_trial_to_csv()


    def _is_task_completed(self) -> bool:
        '''
        Decide whether the grasping task is completed.

        The conditions for replenishment tasks are:
        1. The timer has started
        2. The object list is not empty
        3. All objects have been picked and placed
        '''
        return self._timeline["start"] is not None \
            and self._objects_status \
            and all(obj_info['picked'] for obj_info in self._objects_status.values()) \
            and all(obj_info['placed'] for obj_info in self._objects_status.values())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[Error] No bagfile path provided!")
        print(USAGE_PROMPT_MESSAGE)
        sys.exit(1)
    
    bagfile_path = sys.argv[1]
    task_type = bagfile_path[-7]
    if task_type == 's':
        analyzer = SingleObjectGraspingAnalyzer(bagfile_path=bagfile_path)
    elif task_type == 'm':
        analyzer = MultipleObjectGraspingAnalyzer(bagfile_path=bagfile_path)
    elif task_type == 'c':
        analyzer = ReplenishmentTaskAnalyzer(bagfile_path=bagfile_path)
    else:
        print("[Error] Invalid task type!")
        sys.exit(1)
    analyzer.analyze()