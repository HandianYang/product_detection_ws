from enum import Enum
import sys

from geometry_msgs.msg import Point
import rosbag

HEIGHT_TOLERANCE = 0.15     # [m]
POSITION_TOLERANCE = 0.05  # [m]
PICK_AREA_RADIUS = 0.4  # [m]

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
    LIFT_DISTANCE = 0.05    # [m]

    def __init__(self, bagfile_path: str):
        self.__bag = rosbag.Bag(bagfile_path)

        self.__timeline = {
            'start': None,
            'pick': None,
            'end': None
        }
        
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
    
    def analyze(self):
        '''
        The analyzer to collect data for single object grasping tasks.
        '''
        is_ready_to_record_object_position = False
        bag_start_time = self.__bag.get_start_time()
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
                    self.__timeline['start'] = time.to_sec() - bag_start_time
                    is_ready_to_record_object_position = True
            elif topic == GRIPPER_POSE_TOPIC_NAME:
                self.__gripper_position = msg.pose.position
                if self.__is_gripper_closed and self.__timeline['start'] is not None:
                    if self.__is_object_reached() and not self.__is_object_grasped:
                        self.__timeline['pick'] = time.to_sec() - bag_start_time
                        self.__is_object_grasped = True
                else:
                    self.__is_object_grasped = False

            if self.__is_task_complete():
                self.__timeline['end'] = time.to_sec() - bag_start_time
                break
        
        self.__bag.close()
    
    def __is_object_reached(self) -> bool:
        '''
        Detect whether the gripper is close to the object. 

        Looser tolerance is applied to the Z position difference so that user
        can decide the grasping height.
        '''
        xy_dist = self.get_xy_distance(self.__gripper_position, self.__object_position)
        z_dist = self.get_z_distance(self.__gripper_position, self.__object_position)
        return xy_dist < POSITION_TOLERANCE and z_dist < HEIGHT_TOLERANCE

    def __is_task_complete(self) -> bool:
        '''
        Decide whether the grasping task is completed.

        The conditions are:
        1. The timer has started
        2. The object has been grasped
        3. The object has been lifted up for enough distance
        '''
        return self.__timeline['start'] is not None and self.__is_object_grasped and \
            (self.__gripper_position.z - self.__object_position.z) >= SingleGraspingAnalyzer.LIFT_DISTANCE

    def print_result(self) -> None:
        if None in self.__timeline.values():
            print("[Error] Failed to get experiment result due to the following reason:")
            if self.__timeline['start'] is None:
                print("  - invalid start time")
            if self.__timeline['pick'] is None:
                print("  - invalid pick time")
            if self.__timeline['end'] is None:
                print("  - invalid end time")
            return

        duration = self.__timeline['end'] - self.__timeline['start']
        print(f"================== Experiment result ==================")
        print(f"| Task Completion Time:\t{duration:.3f} sec")
        print(f"|")
        print(f"| Bagfile Timeline:")
        print(f"| -- Begin:\t{self.__timeline['start']:.3f} sec")
        print(f"| -- Pick:\t{self.__timeline['pick']:.3f} sec")
        print(f"| -- Complete:\t{self.__timeline['end']:.3f} sec")
        print(f"=======================================================")
    

class MultipleGraspingAnalyzer:
    def __init__(self, bagfile_path: str):
        self.__bag = rosbag.Bag(bagfile_path)

        self.__gripper_position = Point()
        self.__is_gripper_closed = False
        self.__objects_status = {}  # {id: {label, centroid, grasped}}

        self.__topic_list = [
            GRIPPER_TOPIC_NAME,
            GRIPPER_POSE_TOPIC_NAME,
            JOY_TOPIC_NAME,
            OBJECTS_TOPIC_NAME
        ]

        self.__timeline = {
            'start': None,
            'infer': list(),  # gripper enters pick area (intent inference starts working)
            'picked': list(),   # gripper grasps an object
            'placed': list(),  # gripper places the object
            'end': None
        }

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

    def analyze(self) -> None:
        num_complete_items = 0
        target_id = -1
        is_in_pick_area_now = True
        is_ready_to_record_object_position = False
        bag_start_time = self.__bag.get_start_time()

        for topic, msg, time in self.__bag.read_messages(topics=self.__topic_list):
            if topic == GRIPPER_TOPIC_NAME:
                self.__is_gripper_closed = msg.data
            
            elif topic == OBJECTS_TOPIC_NAME:
                if is_ready_to_record_object_position and msg.objects:
                    self.__objects_status.clear()
                    for index, obj in enumerate(msg.objects):
                        self.__objects_status[index] = {
                            'label': obj.label,
                            'centroid': obj.centroid,
                            'picked': False,
                            'placed': False
                        }
                    is_ready_to_record_object_position = False

            elif topic == JOY_TOPIC_NAME:
                # [NOTE] The user can restart the timer by pressing the stick
                #   button again
                if msg.buttons[2] == 1: # analog stick button
                    self.__timeline['start'] = time.to_sec() - bag_start_time
                    is_ready_to_record_object_position = True

            elif topic == GRIPPER_POSE_TOPIC_NAME:
                self.__gripper_position = msg.pose.position
                if self.__timeline['start'] is None:
                    continue
                
                ### Phase 1: Approach to pick area
                if self.__is_in_pick_area():
                    if not is_in_pick_area_now:
                        # Record new time once only
                        if len(self.__timeline['infer']) == num_complete_items:
                            self.__timeline['infer'].append(time.to_sec() - bag_start_time)
                    is_in_pick_area_now = True
                else:
                    is_in_pick_area_now = False

                ### Phase 2: Approach to the object
                if target_id == -1 and is_in_pick_area_now and self.__is_gripper_closed:
                    candidates = {} # id : xy_dist
                    for obj_id, obj_info in self.__objects_status.items():
                        if not obj_info['picked'] and self.__is_object_reached(obj_info['centroid']):
                            candidates[obj_id] = self.get_xy_distance(self.__gripper_position, obj_info['centroid'])

                    if not candidates:
                        continue
                    target_id = min(candidates, key=candidates.get)
                    self.__objects_status[target_id]['picked'] = True
                    self.__timeline['picked'].append(time.to_sec() - bag_start_time)

                ### Phase 3: Transfer to the place area
                if target_id != -1 and not self.__is_gripper_closed:
                    self.__objects_status[target_id]['placed'] = True
                    self.__timeline['placed'].append(time.to_sec() - bag_start_time)
                    num_complete_items += 1
                    target_id = -1
            
            if self.__is_task_complete():
                self.__timeline['end'] = time.to_sec() - bag_start_time
                break
        
        self.__bag.close()

    def __is_in_pick_area(self) -> bool:
        '''
        Detect whether the gripper is inside the pick area.

        The pick area is defined as the union of the spheres centered at
        centroids of the objects, with radius PICK_AREA_RADIUS.
        '''
        for obj_id, obj_info in self.__objects_status.items():
            obj_dist = self.get_distance(self.__gripper_position, obj_info['centroid'])
            if obj_dist < PICK_AREA_RADIUS:
                return True
        return False

    def __is_object_reached(self, object_position: Point) -> bool:
        '''
        Detect whether the gripper is close to the object. 

        Looser tolerance is applied to the Z position difference so that user
        can decide the grasping height.
        '''
        xy_dist = self.get_xy_distance(self.__gripper_position, object_position)
        z_dist = self.get_z_distance(self.__gripper_position, object_position)
        return xy_dist < POSITION_TOLERANCE and z_dist < HEIGHT_TOLERANCE

    def __is_task_complete(self) -> bool:
        '''
        Decide whether the grasping task is completed.

        The conditions are:
        1. The timer has started
        2. All objects have been picked and placed
        '''
        if not self.__objects_status or self.__timeline['start'] is None:
            return False
        
        all_picked = all(obj_info['picked'] for obj_info in self.__objects_status.values())
        all_placed = all(obj_info['placed'] for obj_info in self.__objects_status.values())
        return all_picked and all_placed

    def print_result(self) -> None:
        if None in self.__timeline.values() or not self.__objects_status:
            print("[Error] Failed to get experiment result due to the following reason:")
            if self.__timeline['start'] is None:
                print("  - invalid start time")
            if self.__timeline['end'] is None:
                print("  - invalid end time")
            if not self.__objects_status:
                print("  - empty recorded objects")
            elif not all(obj_info['picked'] for obj_info in self.__objects_status.values()):
                print("  - not all objects picked")
            elif not all(obj_info['placed'] for obj_info in self.__objects_status.values()):
                print("  - not all objects placed")
            return

        duration = self.__timeline['end'] - self.__timeline['start']
        num_objects = len(self.__objects_status)
        print(f"================== Experiment result ==================")
        print(f"| Task Completion Time:\t\t{duration:.3f} sec")
        print(f"| Average Operation Time:\t{duration / num_objects:.3f} sec")
        print(f"|")
        print(f"| Bagfile Timeline:")
        print(f"| -- Begin:\t{self.__timeline['start']:.3f} sec")
        for index in range(num_objects):
            print(f"| -- Enter pick area:\t{self.__timeline['infer'][index]:.3f} sec")
            print(f"| -- Pick object {index+1}:\t{self.__timeline['picked'][index]:.3f} sec")
            print(f"| -- Place object {index+1}:\t{self.__timeline['placed'][index]:.3f} sec")
        print(f"| -- Complete:\t{self.__timeline['end']:.3f} sec")
        print(f"=======================================================")


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