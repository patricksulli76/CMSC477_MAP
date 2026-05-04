import pupil_apriltags
import cv2
import numpy as np
import time
import traceback
from queue import Empty
from robomaster import robot
from robomaster import camera
import robomaster
import math
import map
import a_star_solver

#map_graph = map.Map(13,11, (1,5),(11,5))
map_graph = map.Map(13,11, (11,5),(1,5))

sq_size = 0.266 # meters


apriltag_to_grid = {
    30: (3.0, 8.5, "L"),
    31: (4.0, 8.5, "R"),
    32: (3.0, 6.5, "L"),
    33: (4.0, 6.5, "R"),
    34: (3.5, 5.0, "D"),
    35: (5.5, 9.0, "D"),
    36: (7.5, 9.0, "D"),
    37: (6.5, 6.0, "U"),
    38: (6.0, 4.5, "L"),
    39: (7.0, 4.5, "R"),
    40: (6.0, 2.5, "L"),
    41: (7.0, 2.5, "R"),
    42: (9.0, 8.5, "L"),
    43: (10.0, 8.5, "R"),
    44: (9.0, 6.5, "L"),
    45: (10.0, 6.5, "R"),
    46: (9.5, 5.0, "D"),
}

for x in range(4, 9):
    map_graph.add_rect(x, 9)
    map_graph.add_obstacle(x, 8)
for y in range(5, 10):
    map_graph.add_rect(3, y)
    map_graph.add_obstacle(3, y)
for y in range(5, 10):
    map_graph.add_rect(9, y)
    map_graph.add_obstacle(9, y)
for y in range(1, 6):
    map_graph.add_rect(6, y)
    map_graph.add_obstacle(6, y)

map_graph.add_rect(1, 5, color='limegreen')
map_graph.add_rect(11, 5, color='orangered')


vel_x, vel_y = 0,0
axis = (1,1)

def set_vel(ep_chassis,curr_pos,target_pos):
    goal_x,goal_y = (target_pos[0]-curr_pos[0]+.5,target_pos[1]-curr_pos[1]-.5)

    k = 0.1


    if(abs(goal_x) < .05):
        vel_x = 0
    else:
        vel_x = k if goal_x > 0 else -k

    if(abs(goal_y) < .05):
        vel_y = 0
    else:
        vel_y = k if goal_y > 0 else -k
    
    print("dists: ", goal_x, goal_y)
    

    if(vel_x == 0 and vel_y == 0):
        return True
    print("Vels: ", vel_x*axis[0], -vel_y*axis[1])
    ep_chassis.drive_speed(x=vel_x*axis[0], y=-vel_y*axis[1], z=0)
    time.sleep(0.1)
    return False

class AprilTagDetector:
    def __init__(self, K, family="tag36h11", threads=2, marker_size_m=0.16):
        self.camera_params = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
        self.marker_size_m = marker_size_m
        self.detector = pupil_apriltags.Detector(family, threads)

    def find_tags(self, frame_gray):
        detections = self.detector.detect(frame_gray, estimate_tag_pose=True,
            camera_params=self.camera_params, tag_size=self.marker_size_m)
        return detections

def get_pose_apriltag_in_camera_frame(detection):
    R_ca = detection.pose_R
    t_ca = detection.pose_t

    return t_ca.flatten(), R_ca

def draw_detections(frame, detections):
    for detection in detections:
        pts = detection.corners.reshape((-1, 1, 2)).astype(np.int32)

        frame = cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

        top_left = tuple(pts[0][0])  # First corner
        top_right = tuple(pts[1][0])  # Second corner
        bottom_right = tuple(pts[2][0])  # Third corner
        bottom_left = tuple(pts[3][0])  # Fourth corner
        cv2.line(frame, top_left, bottom_right, color=(0, 0, 255), thickness=2)
        cv2.line(frame, top_right, bottom_left, color=(0, 0, 255), thickness=2)

        tag_id = str(detection.tag_id)
        # Position the text slightly above the top-left corner
        text_position = (top_left[0], top_left[1] - 10)
        
        cv2.putText(frame, 
                    f"ID: {tag_id}", 
                    text_position, 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8,              # Font scale (size)
                    (0, 255, 0),      # Color (BGR format - Green)
                    2)                # Thickness

def detect_tag_loop(ep_robot, ep_chassis, ep_camera, apriltag):
    # map_path = a_star_solver.a_star(map_graph.graph,map_graph.start,map_graph.finish)
    # localizing = 0
    # curr_point = 0
    # map_graph.add_edge(map_graph.start[0], map_graph.start[1], map_path[0][0], map_path[0][1])
    # for i in range(len(map_path)):
    #     map_graph.add_edge(map_path[i][0], map_path[i][1], map_path[i+1][0], map_path[i+1][1])

    stage = 0
    while True:
        try:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        except Empty:
            time.sleep(0.001)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray.astype(np.uint8)
        centering_boundry = 10 # degrees

        detections = apriltag.find_tags(gray)
        if stage == 0:
            if turn_to_tag(detections, ep_chassis, 32):
                stage = 1
                print("Stage 0 completed")
        elif stage == 1:
            if move_to_tag(detections, ep_chassis, 38, "RIGHT"):
                stage = 2
                print("Stage 1 completed")
                ep_chassis.move(x=0, y=sq_size*0.75, z=0, xy_speed=0.2).wait_for_completed()
        elif stage == 2:
            print("Starting stage 2")
            if approach_tag(detections, ep_chassis, 38, sq_size*1.5):
                stage = 3
                print("Stage 2 completed")
            else:
                print("Failed to approach tag")
        elif stage == 3:
            if turn_to_tag(detections, ep_chassis, 35):
                stage = 4 
                print("Stage 3 completed")
        elif stage == 4:
            if approach_tag(detections, ep_chassis, 35, 2.5*sq_size):
                stage = 5
                print("Stage 4 completed")
        elif stage == 5:
            if turn_to_tag(detections, ep_chassis, 44):
                stage = 6
                print("Stage 5 completed")
        elif stage == 6:
            if approach_tag(detections, ep_chassis, 44, sq_size*1.75):
                stage = 7
                ep_chassis.move(x=0, y=0, z=190, z_speed=45).wait_for_completed()
                print("Stage 6 completed")
        elif stage == 7:
            if move_to_tag(detections, ep_chassis, 39, "LEFT"):
                stage = 8
                print("Stage 7 completed")
        elif stage == 8:
            if disengage_tag(detections, ep_chassis, 39, 4.5*sq_size):
                stage = 9
                print("Stage 8 completed")
        elif stage == 9:
            if move_to_finish(detections, ep_chassis, 45, "RIGHT"):
                return
                print("Finished")
            
            

        # draw_detections(img, detections)
        # cv2.imshow("img", img)
        # if cv2.waitKey(1) == ord('q'):
        #     break
def is_centered(t_ca):
    offset = 10 # degrees
    theta = math.degrees(math.atan2(t_ca[2], t_ca[0]))
    if theta <= 90+offset and theta >= 90-offset:
        return 0
    elif theta > 90+offset:
        return 1
    else:
        return -1

def stage_0(detections, ep_chassis):
    detection = next((d for d in detections if d.tag_id == 32), None)
    if detection is not None:
        t_ca, R_ca = get_pose_apriltag_in_camera_frame(detection)
        if is_centered(t_ca) == 0:
            return True
        elif is_centered(t_ca) == 1:
            ep_chassis.move(x=0, y=0, z=5, z_speed=45).wait_for_completed()
        else:
            ep_chassis.move(x=0, y=0, z=-5, z_speed=45).wait_for_completed()
    return False


def stage_1(detections, ep_chassis):
    detection = next((d for d in detections if d.tag_id == 38), None)
    if detection is not None:
        t_ca, R_ca = get_pose_apriltag_in_camera_frame(detection)
        if is_centered(t_ca) == 0:
            return True
        elif is_centered(t_ca) == 1:
            ep_chassis.move(x=0, y=-0.2, z=0, xy_speed=0.2).wait_for_completed()
    else:
        ep_chassis.move(x=0, y=-0.2, z=0, xy_speed=0.2).wait_for_completed()
    return False

def stage_2(detections, ep_chassis):
    detection = next((d for d in detections if d.tag_id == 38), None)
    if detection is not None:
        t_ca, R_ca = get_pose_apriltag_in_camera_frame(detection)
        distance = np.linalg.norm(t_ca)
        if distance < sq_size:
            return True
        else:
            ep_chassis.move(x=0.1, y=0, z=0, xy_speed=0.2).wait_for_completed()
    else:
        ep_chassis.move(x=-0.1, y=0, z=0, xy_speed=0.2).wait_for_completed()
    return False

def stage_3(detections, ep_chassis):
    detection = next((d for d in detections if d.tag_id == 35), None)
    if detection is not None:
        t_ca, R_ca = get_pose_apriltag_in_camera_frame(detection)
        if is_centered(t_ca) == 0:
            return True
        elif is_centered(t_ca) == 1:
            ep_chassis.move(x=0, y=0, z=5, z_speed=45).wait_for_completed()
        else:
            ep_chassis.move(x=0, y=0, z=-5, z_speed=45).wait_for_completed()
    return False

def stage_4(detections, ep_chassis):
    detection = next((d for d in detections if d.tag_id == 35), None)
    if detection is not None:
        t_ca, R_ca = get_pose_apriltag_in_camera_frame(detection)
        distance = np.linalg.norm(t_ca)
        if distance < sq_size:
            return True
        else:
            ep_chassis.move(x=0.1, y=0, z=0, xy_speed=0.2).wait_for_completed()
    else:
        ep_chassis.move(x=-0.1, y=0, z=0, xy_speed=0.2).wait_for_completed()
    return False

def approach_tag(detections, ep_chassis, id,distance_offset):
    detection = next((d for d in detections if d.tag_id == id), None)
    if detection is not None:
        t_ca, R_ca = get_pose_apriltag_in_camera_frame(detection)
        distance = np.linalg.norm(t_ca)
        print("Distance: ", distance)
        if distance <= distance_offset:
            return True
        else:
            print("Moving closer to tag")
            ep_chassis.move(x=0.1, y=0, z=0, xy_speed=0.2).wait_for_completed()
    else:
        print("No detection found")
        ep_chassis.move(x=0.1, y=0, z=0, xy_speed=0.2).wait_for_completed()
        return True
    return False

def turn_to_tag(detections, ep_chassis, id):
    detection = next((d for d in detections if d.tag_id == id), None)
    if detection is not None:
        t_ca, R_ca = get_pose_apriltag_in_camera_frame(detection)
        if is_centered(t_ca) == 0:
            return True
        elif is_centered(t_ca) == 1:
            ep_chassis.move(x=0, y=0, z=5, z_speed=45).wait_for_completed()
        else:
            ep_chassis.move(x=0, y=0, z=-5, z_speed=45).wait_for_completed()
    else:
        print("No detection found")
        ep_chassis.move(x=0, y=0, z=5, z_speed=20).wait_for_completed()
    return False

def move_to_tag(detections, ep_chassis, id, direction):
    if direction == "LEFT":
        direction = -2
    elif direction == "RIGHT":
        direction = 2
    else:
        direction = 0
    print("Moving to tag: ", id, "in direction: ", direction)
    detection = next((d for d in detections if d.tag_id == id), None)
    if detection is not None:
        t_ca, R_ca = get_pose_apriltag_in_camera_frame(detection)
        if is_centered(t_ca) == 0:
            return True
        elif is_centered(t_ca) == 1:
            ep_chassis.move(x=0, y=0.1*direction, z=0, xy_speed=0.2*abs(direction)).wait_for_completed()
    else:
        ep_chassis.move(x=-0.1, y=0.1*direction, z=0, xy_speed=0.2*abs(direction)).wait_for_completed()
    return False

def disengage_tag(detections, ep_chassis, id,distance_offset):
    detection = next((d for d in detections if d.tag_id == id), None)
    if detection is not None:
        t_ca, R_ca = get_pose_apriltag_in_camera_frame(detection)
        distance = np.linalg.norm(t_ca)
        if distance > distance_offset:
            return True
        else:
            ep_chassis.move(x=-0.1, y=0, z=0, xy_speed=0.2).wait_for_completed()
    else:
        ep_chassis.move(x=0.1, y=0, z=0, xy_speed=0.2).wait_for_completed()
        return True
    return False
    
def move_to_finish(detections, ep_chassis, id, direction):
    if direction == "LEFT":
        direction = -2
    elif direction == "RIGHT":
        direction = 2
    else:
        direction = 0
    print("Moving to tag: ", id, "in direction: ", direction)
    detection = next((d for d in detections if d.tag_id == id), None)
    if detection is not None:
        t_ca, R_ca = get_pose_apriltag_in_camera_frame(detection)
        if is_centered(t_ca) == 0:
            return True
        elif is_centered(t_ca) == 1:
            ep_chassis.move(x=0, y=0.25*direction, z=0, xy_speed=0.2*abs(direction)).wait_for_completed()
    else:
        ep_chassis.move(x=0, y=0.25*direction, z=0, xy_speed=0.2*abs(direction)).wait_for_completed()
    return False


if __name__ == '__main__':
    # More legible printing from numpy.
    np.set_printoptions(precision=3, suppress=True, linewidth=120)

    robomaster.config.ROBOT_IP_STR = "192.168.50.113"
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="sta", sn="3JKCH8800100YN")
    ep_chassis = ep_robot.chassis
    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)

    K = np.array([[314, 0, 320], [0, 314, 180], [0, 0, 1]]) # Camera focal length and center pixel
    marker_size_m = 0.153 # Size of the AprilTag in meters
    apriltag = AprilTagDetector(K, threads=2, marker_size_m=marker_size_m)
    
    try:
        detect_tag_loop(ep_robot, ep_chassis, ep_camera, apriltag)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(traceback.format_exc())
    finally:
        print('Waiting for robomaster shutdown')
        ep_chassis.drive_speed(x=0, y=0, z=0, timeout=1)
        time.sleep(1)
        ep_camera.stop_video_stream()
        ep_robot.close()
