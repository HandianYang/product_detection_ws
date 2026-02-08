from enum import Enum
import sys

from geometry_msgs.msg import Point
import rosbag

POSITION_TOLERANCE = 0.05  # [m]
HEIGHT_TOLERANCE = 0.15     # [m]

GRIPPER_TOPIC_NAME = '/gripper/cmd_gripper'
GRIPPER_POSE_TOPIC_NAME = '/marslite_control/gripper_pose'
JOY_TOPIC_NAME = '/vr_controller/joy'
OBJECTS_TOPIC_NAME = '/marslite_control/objects_with_belief'

USAGE_PROMPT_MESSAGE = "\
  Usage: python3 analyze_bagfile.py <user>_<ctrl><task><trial>[_rX]\n\
    <user>: 01, 02, ..., 99\n\
    <ctrl>: m (manual control), s (shared control)\n\
    <task>: s (single item), m (multiple items), c (complete task)\n\
    <trial>: 01, 02, ..., 99\n\
    [_rX]: (optional) X-th repeat of the same trial (e.g., _r1, _r2)\n\n\
  Example: \n\
    python3 analyze_bagfile.py 01_ms01\n\
    python3 analyze_bagfile.py 02_sm03_r2\n"


class SingleGraspingAnalyzer:
    # START_DELAY_TIME = 0.5  # [s]
    LIFT_DISTANCE = 0.05    # [m]

    def __init__(self, bagfile_path: str):
        self.__bagfile_path = bagfile_path
        self.__bag = rosbag.Bag(bagfile_path)

        self.__start_time = None
        self.__end_time = None
        
        self.__gripper_position = Point()
        self.__is_gripper_closed = False
        self.__object_position = Point()
        self.__is_object_grasped = False

        self.__topic_list = [
            GRIPPER_TOPIC_NAME,
            GRIPPER_POSE_TOPIC_NAME,
            JOY_TOPIC_NAME,
            OBJECTS_TOPIC_NAME
        ]
    
    def analyze(self):
        '''
        The analyzer to collect data for single object grasping tasks.
        '''
        is_ready_to_record_object_position = False
        # print(f"Analyzing {self.__bagfile_path} ...")

        for topic, msg, time in self.__bag.read_messages(topics=self.__topic_list):    
            if topic == GRIPPER_TOPIC_NAME:
                self.__is_gripper_closed = msg.data
            elif topic == OBJECTS_TOPIC_NAME: 
                if msg.objects and is_ready_to_record_object_position:
                    self.__object_position = msg.objects[0].centroid
                    is_ready_to_record_object_position = False
            elif topic == JOY_TOPIC_NAME:
                # [NOTE] The user can restart the timer by pressing the stick
                #   button again
                if msg.buttons[2] == 1: # analog stick button
                    self.__start_time = time.to_sec()
                    is_ready_to_record_object_position = True
                    print(f"Record starts at {time.to_sec():.3f}")
            elif topic == GRIPPER_POSE_TOPIC_NAME:
                self.__gripper_position = msg.pose.position
                if self.__is_gripper_closed and self.__start_time is not None:
                    if self.__is_object_reached() and not self.__is_object_grasped:
                        self.__is_object_grasped = True
                        print(f"Object grasped at {time.to_sec():.3f}")
                else:
                    self.__is_object_grasped = False

            if self.__is_task_complete():
                self.__end_time = time.to_sec()
                print(f"Record ends at {time.to_sec():.3f}")
                break

        self.__bag.close()
    
    def __is_object_reached(self) -> bool:
        '''
        Detect whether the gripper is close to the object. 

        Looser tolerance is applied to the Z position difference so that user
        can decide the grasping height.
        '''
        dx = self.__gripper_position.x - self.__object_position.x
        dy = self.__gripper_position.y - self.__object_position.y
        dz = self.__gripper_position.z - self.__object_position.z
        return (dx * dx + dy * dy) ** 0.5 < POSITION_TOLERANCE and abs(dz) < HEIGHT_TOLERANCE

    def __is_task_complete(self) -> bool:
        '''
        Decide whether the grasping task is completed.

        The conditions are:
        1. The timer has started
        2. The object has been grasped
        3. The object has been lifted up for enough distance
        '''
        return self.__start_time is not None and self.__is_object_grasped and \
            (self.__gripper_position.z - self.__object_position.z) >= SingleGraspingAnalyzer.LIFT_DISTANCE

    def print_result(self) -> None:
        if self.__start_time is None or self.__end_time is None:
            print("[Error] Failed to get experiment result due to the following reason:")
            if self.__start_time is None:
                print("    - invalid start time")
            if self.__end_time is None:
                print("    - invalid end time")
            return

        duration = self.__end_time - self.__start_time
        print(f"================== Experiment result ==================")
        print(f"| Task Completion Time (TCT) : {duration:.3f} sec")
        print(f"=======================================================")
    

class MultipleGraspingAnalyzer:
    # START_DELAY_TIME = 0.5  # [s]

    def __init__(self, bagfile_path: str):
        self.__bagfile_path = bagfile_path
        self.__bag = rosbag.Bag(bagfile_path)

        self.__start_time = None
        self.__end_time = None

        self.__gripper_position = Point()
        self.__is_gripper_closed = False
        self.__objects_status = {}  # {id: {label, centroid, grasped}}

        self.__topic_list = [
            GRIPPER_TOPIC_NAME,
            GRIPPER_POSE_TOPIC_NAME,
            JOY_TOPIC_NAME,
            OBJECTS_TOPIC_NAME
        ]

    def analyze(self) -> None:
        is_ready_to_record_object_position = False

        # print(f"Analyzing {self.__bagfile_path} ...")
        for topic, msg, time in self.__bag.read_messages(topics=self.__topic_list):
            if topic == GRIPPER_TOPIC_NAME:
                self.__is_gripper_closed = msg.data
            
            elif topic == OBJECTS_TOPIC_NAME:
                if is_ready_to_record_object_position:
                    self.__objects_status.clear()
                    for index, obj in enumerate(msg.objects):
                        self.__objects_status[index] = {
                            'label': obj.label,
                            'centroid': obj.centroid,
                            'grasped': False
                        }
                    is_ready_to_record_object_position = False

            elif topic == JOY_TOPIC_NAME:
                # [NOTE] The user can restart the timer by pressing the stick
                #   button again
                if msg.buttons[2] == 1: # analog stick button
                    self.__start_time = time.to_sec()
                    is_ready_to_record_object_position = True
                    print(f"Record starts at {time.to_sec():.3f}")

            elif topic == GRIPPER_POSE_TOPIC_NAME:
                self.__gripper_position = msg.pose.position
                if self.__is_gripper_closed and self.__start_time is not None:
                    for obj_id, obj_info in self.__objects_status.items():
                        if not obj_info['grasped'] and self.__is_object_reached(obj_info['centroid']):
                            obj_info['grasped'] = True
                            print(f"Object [{obj_info['label']}] (ID: {obj_id}) grasped at {time.to_sec():.3f}")
            
            if self.__is_task_complete():
                self.__end_time = time.to_sec()
                print(f"Record ends at {time.to_sec():.3f}")
                break

        self.__bag.close()

    def __is_object_reached(self, object_position: Point) -> bool:
        '''
        Detect whether the gripper is close to the object. 

        Looser tolerance is applied to the Z position difference so that user
        can decide the grasping height.
        '''
        dx = self.__gripper_position.x - object_position.x
        dy = self.__gripper_position.y - object_position.y
        dz = self.__gripper_position.z - object_position.z
        return (dx * dx + dy * dy) ** 0.5 < POSITION_TOLERANCE and abs(dz) < HEIGHT_TOLERANCE

    def __is_task_complete(self) -> bool:
        '''
        Decide whether the grasping task is completed.

        The conditions are:
        1. The timer has started
        2. All objects have been grasped
        3. The gripper is opened (the last grasped object is released)
        '''
        if not self.__objects_status:
            return False
        
        all_grasped = all(obj_info['grasped'] for obj_info in self.__objects_status.values())
        return self.__start_time is not None and all_grasped and not self.__is_gripper_closed

    def print_result(self) -> None:
        if self.__start_time is None or self.__end_time is None:
            print("[Error] Failed to get experiment result due to the following reason:")
            if self.__start_time is None:
                print("    - invalid start time")
            if self.__end_time is None:
                print("    - invalid end time")
            return
        
        if not self.__objects_status:
            print("[Error] Failed to get experiment result due to the following reason:")
            print("    - empty recorded objects")
            return

        duration = self.__end_time - self.__start_time
        num_objects = len(self.__objects_status)
        print(f"================== Experiment result ==================")
        print(f"| Task Completion Time (TCT) : {duration:.3f} sec")
        print(f"| Average Grasping Time (AGT)   : {duration / num_objects:.3f} sec")
        print(f"=======================================================")
        print("[Error] Failed to get experiment result due to invalid start and end time!")



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[Error] No bagfile path provided!")
        print(USAGE_PROMPT_MESSAGE)
        sys.exit(1)
    
    bagfile_path = sys.argv[1]
    print(f"  [Bagfile name] {bagfile_path}")
    # file format: <user>_<ctrl><task><trial>[_rX].bag
    # exmaples:
    # (1) bag with no repeated trial: 01_ms01.bag
    # (2) bag with repeated trial: 02_sm03_r2.bag
    if bagfile_path[-6] == 'r':
        ctrl_type = bagfile_path[-11]
        task_type = bagfile_path[-10]
    else:
        ctrl_type = bagfile_path[-8]
        task_type = bagfile_path[-7]

    if ctrl_type == 'm':
        print("  [Control type] manual control")
    elif ctrl_type == 's':
        print("  [Control type] shared control")
    else:
        print("  [Error] Invalid control type!")
        sys.exit(1)
    
    if task_type == 's':
        print("  [Task type]    single object grasping")
        analyzer = SingleGraspingAnalyzer(bagfile_path=bagfile_path)
    elif task_type == 'm':
        print("  [Task type]    multiple object grasping")
        analyzer = MultipleGraspingAnalyzer(bagfile_path=bagfile_path)
    elif task_type == 'c':
        print("  [Task type]    complete replenishment task")
    else:
        print("  [Error] Invalid task type!")
        sys.exit(1)
    
    analyzer.analyze()
    analyzer.print_result()