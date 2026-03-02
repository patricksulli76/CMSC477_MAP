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

map_graph = map.Map(13,11)

sq_size = 0.266 # meters


apriltag_to_grid = {
    30: (3, 8, "L"),
    31: (3, 8, "R"),
    32: (3, 6, "L"),
    33: (3, 6, "R"),
    34: (3, 5, "D"),
    35: (5, 9, "D"),
    36: (7, 9, "D"),
    37: (6, 5, "U"),
    38: (6, 4, "L"),
    39: (6, 4, "R"),
    40: (6, 2, "L"),
    41: (6, 2, "R"),
    42: (9, 8, "L"),
    43: (9, 8, "R"),
    44: (9, 6, "L"),
    45: (9, 6, "R"),
    46: (9, 5, "D"),
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

def detect_tag_loop(ep_robot, ep_chassis, ep_camera, apriltag):
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
            for detection in detections:
                t_ca, R_ca = get_pose_apriltag_in_camera_frame(detection)
                id = detection.tag_id
                print('t_ca', t_ca)
                print('R_ca', R_ca)
                scaled = t_ca / sq_size
                orientation = apriltag_to_grid[id][2]

                if orientation == "D":
                    x_sum += apriltag_to_grid[id][0] - scaled[2]
                    y_sum += apriltag_to_grid[id][1] + scaled[0]
                elif orientation == "U":
                    x_sum += apriltag_to_grid[id][0] + scaled[2]
                    y_sum += apriltag_to_grid[id][1] - scaled[0]
                elif orientation == "R":
                    x_sum += apriltag_to_grid[id][0] + scaled[0]
                    y_sum += apriltag_to_grid[id][1] + scaled[2]
                elif orientation == "L":
                    x_sum += apriltag_to_grid[id][0] - scaled[0]
                    y_sum += apriltag_to_grid[id][1] - scaled[2]
            x_avg = x_sum / len(detections)
            y_avg = y_sum / len(detections)
            print('x_avg', x_avg)
            print('y_avg', y_avg)
            map_graph.remove_last_point()
            map_graph.add_point(x_avg, y_avg)
            map_graph.show_graph()

        draw_detections(img, detections)
        cv2.imshow("img", img)
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
