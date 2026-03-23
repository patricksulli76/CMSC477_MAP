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

map_graph = map.Map(13,11, (1,6),(11,6))
#map_graph = map.Map(13,11, (11,5),(1,5))
orientation = "L"

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

    x_target = target_pos[0]-curr_pos[0]
    y_target = target_pos[1]-curr_pos[1]
    dist_x,dist_y = (0,0)

    if orientation == "L":
        dist_x = x_target
        dist_y = y_target
    elif orientation == "D":
        dist_x = -y_target
        dist_y = x_target
    elif orientation == "R":
        dist_x = -x_target
        dist_y = -y_target
    elif orientation == "U":
        dist_x = y_target
        dist_y = -x_target

    k = 0.1


    if(abs(dist_x) < .05):
        vel_x = 0
    else:
        vel_x = k if dist_x > 0 else -k

    if(abs(dist_y) < .05):
        vel_y = 0
    else:
        vel_y = k if dist_y > 0 else -k
    
    print("distance to goal: ", dist_x, dist_y)
    

    if(vel_x == 0 and vel_y == 0):
        return True
    print("Vels: ", vel_x*axis[0], -vel_y*axis[1])

    if orientation == "L":
        ep_chassis.drive_speed(x=vel_x, y=-vel_y, z=0)
    elif orientation == "D":
        ep_chassis.drive_speed(x=vel_y, y=vel_x, z=0)
    elif orientation == "R":
        ep_chassis.drive_speed(x=-vel_x, y=vel_y, z=0)
    elif orientation == "U":
        ep_chassis.drive_speed(x=-vel_y, y=vel_x, z=0)
    #ep_chassis.drive_speed(x=vel_x*axis[0], y=-vel_y*axis[1], z=0)
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
    map_path = a_star_solver.a_star(map_graph.graph,map_graph.start,map_graph.finish)
    for i in range(len(map_path)-1):
        map_graph.add_edge(map_path[i][0], map_path[i][1], map_path[i+1][0], map_path[i+1][1])
    localizing = 0
    curr_point = 0
    while True:
        try:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        except Empty:
            time.sleep(0.001)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray.astype(np.uint8)


        detections = apriltag.find_tags(gray)
        if len(detections) > 0:
            x_sum = 0
            y_sum = 0
            total_weight = 1e-6
            for detection in detections:
                if detection.decision_margin < 25: 
                    continue
                try:
                    t_ca, R_ca = get_pose_apriltag_in_camera_frame(detection)
                    id = detection.tag_id
                    # print("ID: ", id)
                    # print('t_ca', t_ca)
                    # print('R_ca', R_ca)
                    
                    scaled = t_ca / sq_size
                    distance = np.linalg.norm(t_ca)
                    weight = 1.0 / (distance**2 + 1e-6)
                    # print('Scaled', scaled)
                    orientation = apriltag_to_grid[id][2]

                    
                    if orientation == "D":
                        y_val = apriltag_to_grid[id][1] - scaled[2]
                        x_val = apriltag_to_grid[id][0] - scaled[0]
                    elif orientation == "U":
                        y_val = apriltag_to_grid[id][1] + scaled[2]
                        x_val = apriltag_to_grid[id][0] + scaled[0]
                    elif orientation == "R":
                        y_val = apriltag_to_grid[id][1] - scaled[0]
                        x_val = apriltag_to_grid[id][0] + scaled[2]
                    elif orientation == "L":
                        y_val = apriltag_to_grid[id][1] + scaled[0]
                        x_val = apriltag_to_grid[id][0] - scaled[2]
                    x_sum += x_val * weight
                    y_sum += y_val * weight
                    total_weight += weight
                except:
                    continue
            
            x_avg = x_sum / total_weight
            y_avg = y_sum / total_weight
            print('x_avg, y_avg:', x_avg, y_avg)
            map_graph.remove_last_point()
            map_graph.add_point(x_avg, y_avg)
            map_graph.show_graph()
            print('goal_x, goal_y:', map_path[curr_point][0], map_path[curr_point][1])
            if (localizing != 0):
                localizing += 90
                ep_chassis.move(x=0, y=0, z=90, z_speed=45).wait_for_completed()
                if localizing == 360:
                    localizing = 0
            if(curr_point < len(map_path) and localizing == 0):
                if set_vel(ep_chassis,(x_avg,y_avg),map_path[curr_point]):
                    curr_point+=1
        else:

            #localizing += 90
            # if (vel_x > 0):
            #     ep_chassis.move(x=-0.1, y=0, z=0, xy_speed=0.2).wait_for_completed()
            # elif (vel_x > 0):
            #     ep_chassis.move(x=-0.1, y=0, z=0, xy_speed=0.2).wait_for_completed()
            # if (vel_y > 0):
            #     ep_chassis.move(x=0, y=.1, z=0, xy_speed=0.2).wait_for_completed()
            # elif (vel_y > 0):
            #     ep_chassis.move(x=0, y=-.1, z=0, xy_speed=0.2).wait_for_completed()
            ep_chassis.move(x=0, y=0, z=90, z_speed=45).wait_for_completed()
            

        draw_detections(img, detections)
        # cv2.imshow("img", img)
        if cv2.waitKey(1) == ord('q'):
            break

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
