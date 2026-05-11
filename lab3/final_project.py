from queue import Empty
import traceback
import cv2
import time
import numpy as np
from ultralytics import YOLO
from robomaster import robot
import pupil_apriltags

######### CONSTANTS #########
SMALL_BLOCK_SIZE = (0.06, 0.1) # w,h
LARGE_BLOCK_SIZE = (0.06, 0.185) # w,h
CONE_SIZE = (0.13, 0.24) # w,h
BOX_SIZE = (0.26, 0.26) # w,h

camera_matrix = np.array([[314, 0, 320], [0, 314, 180], [0, 0, 1]])

BATTERY_CHARGE = 0.6
SMALL_BLOCK_CHARGE = 0.3
LARGE_BLOCK_CHARGE = 0.4

ARUCO_SIZE = 0.16

CHARGINGTAG_ID1 = 34
CHARGINGTAG_ID2 = 38

SMALL_GOAL_ID1 = 11
SMALL_GOAL_ID2 = 41

LARGE_GOAL_ID1 = 19
LARGE_GOAL_ID2 = 45
#############################


def solve_pnp_rectangle_large(corners_2d, dist_coeffs=None):
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5)

    half_w = LARGE_BLOCK_SIZE[0] / 2.0
    half_h = LARGE_BLOCK_SIZE[1] / 2.0

    obj_pts = np.array([
        [-half_w,  half_h, 0],
        [ half_w,  half_h, 0],
        [ half_w, -half_h, 0],
        [-half_w, -half_h, 0],
    ], dtype=np.float64)

    img_pts = corners_2d.astype(np.float64).reshape(4, 1, 2)

    success, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE
    )

    return tvec

def solve_pnp_rectangle_small(corners_2d, dist_coeffs=None):
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5)

    half_w = SMALL_BLOCK_SIZE[0] / 2.0
    half_h = SMALL_BLOCK_SIZE[1] / 2.0

    obj_pts = np.array([
        [-half_w,  half_h, 0],
        [ half_w,  half_h, 0],
        [ half_w, -half_h, 0],
        [-half_w, -half_h, 0],
    ], dtype=np.float64)

    img_pts = corners_2d.astype(np.float64).reshape(4, 1, 2)

    success, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE
    )

    return 


def solve_pnp_rectangle_box(corners_2d, dist_coeffs=None):
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5)

    half_w = BOX_SIZE[0] / 2.0
    half_h = BOX_SIZE[1] / 2.0

    obj_pts = np.array([
        [-half_w,  half_h, 0],
        [ half_w,  half_h, 0],
        [ half_w, -half_h, 0],
        [-half_w, -half_h, 0],
    ], dtype=np.float64)

    img_pts = corners_2d.astype(np.float64).reshape(4, 1, 2)

    success, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE
    )

    return tvec

def solve_pnp_rectangle_cone(corners_2d, dist_coeffs=None):
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5)

    half_w = CONE_SIZE[0] / 2.0
    half_h = CONE_SIZE[1] / 2.0

    obj_pts = np.array([
        [-half_w,  half_h, 0],
        [ half_w,  half_h, 0],
        [ half_w, -half_h, 0],
        [-half_w, -half_h, 0],
    ], dtype=np.float64)

    img_pts = corners_2d.astype(np.float64).reshape(4, 1, 2)

    success, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE
    )

    return 

########### ROBOT HELPER FUNCTIONS ###########

def execute_pick_sequence(robot_obj):
    arm = robot_obj.robotic_arm
    gripper = robot_obj.gripper
    print(">>> EXECUTING GROUND-LEVEL GRAB")
    arm.moveto(x=185, y=-70).wait_for_completed()
    gripper.close(power=50)
    time.sleep(1.5)
    gripper.pause() 
    arm.moveto(x=120, y=100).wait_for_completed()
    print(">>> OBJECT SECURED")

def execute_delivery_sequence(robot_obj, size):
    chassis = robot_obj.chassis
    arm = robot_obj.robotic_arm
    gripper = robot_obj.gripper
    camera = robot_obj.camera

    print(">>> ROTATING 90 DEGREES RIGHT TO DEPOSIT")
    chassis.move(z=-180).wait_for_completed()
    
    print(">>> MOVING 0.5M TO DELIVERY POINT")
    chassis.move(x=0.5).wait_for_completed()
    
    print(">>> LOWERING AND RELEASING")
    arm.moveto(x=185, y=-70).wait_for_completed()
    gripper.open(power=50)
    time.sleep(1.0)

    print(">>> RETURNING TO SEARCH AREA")
    chassis.move(x=-1).wait_for_completed() 
    arm.moveto(x=180, y=-10).wait_for_completed()

    flush_camera(robot_obj)
    
    print(">>> READY FOR NEXT TARGET")

    
def flush_camera(robot_obj):
    camera = robot_obj.camera
    chassis = robot_obj.chassis
    chassis.drive_speed(x=0, y=0, z=0)
    print(">>> FLUSHING CAMERA BUFFER...")
    time.sleep(0.5) 
    for _ in range(10):
        camera.read_cv2_image(strategy="newest")

def pixel_to_camera_coords(u, v, Z, camera_matrix):
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return X, Y



############# DETECTING NEAREST OBJECT ############
def _detect_nearest(results, class_id, solve_pnp_fn, label, annotated_frame=None):
    """
    Generic helper to detect the nearest object of a given class.

    Args:
        results: YOLO results object
        class_id: YOLO class index to filter
        solve_pnp_fn: function to call for pose estimation
        label: string label for printing/drawing
        annotated_frame: optional frame to draw detections on

    Returns:
        nearest_tvec, nearest_center, nearest_dist, annotated_frame
    """
    nearest_tvec = None
    nearest_center = None
    nearest_dist = float('inf')

    if len(results) == 0:
        return nearest_tvec, nearest_center, nearest_dist, annotated_frame

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return nearest_tvec, nearest_center, nearest_dist, annotated_frame

    xyxy_array = boxes.xyxy.cpu().numpy()
    cls_array  = boxes.cls.cpu().numpy()
    conf_array = boxes.conf.cpu().numpy()

    for i, box in enumerate(xyxy_array):
        if int(cls_array[i]) != class_id:
            continue

        x1, y1, x2, y2 = box.astype(int)

        corners_2d = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ], dtype=np.float64)

        tvec = solve_pnp_fn(corners_2d)
        if tvec is None:
            continue

        depth = tvec[2][0]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        print(f"{label} {i}: depth={depth:.2f}m center=({cx:.0f}, {cy:.0f}) conf={conf_array[i]:.2f}")

        if annotated_frame is not None:
            color = (0, 255, 255) if depth < nearest_dist else (0, 165, 255)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, f"{label} {depth:.2f}m",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if depth < nearest_dist:
            nearest_dist = depth
            nearest_tvec = tvec
            nearest_center = (cx, cy)

    if nearest_tvec is not None:
        print(f"Nearest {label}: depth={nearest_dist:.2f}m center={nearest_center}")

    return nearest_tvec, nearest_center, nearest_dist, annotated_frame


def detect_nearest_cone(results, annotated_frame=None):
    return _detect_nearest(results, class_id=5, solve_pnp_fn=solve_pnp_rectangle_cone,
                           label="Cone", annotated_frame=annotated_frame)

def detect_nearest_small_block(results, annotated_frame=None):
    return _detect_nearest(results, class_id=0, solve_pnp_fn=solve_pnp_rectangle_small,
                           label="SmallBlock", annotated_frame=annotated_frame)

def detect_nearest_large_block(results, annotated_frame=None):
    return _detect_nearest(results, class_id=3, solve_pnp_fn=solve_pnp_rectangle_large,
                           label="LargeBlock", annotated_frame=annotated_frame)

def detect_nearest_box(results, annotated_frame=None):
    return _detect_nearest(results, class_id=6, solve_pnp_fn=solve_pnp_rectangle_box,
                           label="Box", annotated_frame=annotated_frame)



############# Aruco Stuff ############

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


###################################################
states = {
    0: "SEARCHING FOR BLOCK",
    1: "GETTING BLOCK",
    2: "DELIVERING BLOCK TO GOAL",
    3: "NAVIGATE TO CHARGING STATION",
}

curr_state = 0
MODEL_PATH = "best.pt"
#MODEL_PATH = "better.pt"
CENTER_X = 320    
CENTER_Y = 180           
VISIBLE_GRAB_THRESHOLD = 320
BLIND_SPOT_Y = 410     
DRIVE_SPEED = 0.04     
SEARCH_TIMEOUT = 15.0 

def detect_tag_loop(ep_robot, ep_chassis, ep_camera, apriltag):
    while True:
        
        try:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        except Empty:
            time.sleep(0.001)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray.astype(np.uint8)

        model = YOLO(MODEL_PATH)
        results = model(img, conf=0.5, verbose=False)


        detections = apriltag.find_tags(gray)
        min_weight = 100000
        target_id = 0

#         - block
# - robot
# - paper
# - yellow
# - purple
# - cone
# - fabric_box

        robots = results[1]
        cones = results[5]
        boxes = results[6]

        if curr_state != 1:
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
                        
                        scaled = t_ca / ARUCO_SIZE
                        distance = np.linalg.norm(t_ca)
                        weight = 1.0 / (distance**2 + 1e-6)
                        if weight < min_weight:
                            min_weight = weight
                            target_id = id
                        # print('Scaled', scaled)
                        x_val, y_val = pixel_to_camera_coords(detection.center[0], detection.center[1], t_ca[2], camera_matrix)
                        x_sum += x_val * weight
                        y_sum += y_val * weight
                        total_weight += weight

                    except:
                        continue
                

                
                

            draw_detections(img, detections)
            # cv2.imshow("img", img)
            if cv2.waitKey(1) == ord('q'):
                break
        if curr_state == 1:
            pass
        if curr_state == 2:
            pass
        if curr_state == 3:
            pass

if __name__ == '__main__':
    # More legible printing from numpy.
    np.set_printoptions(precision=3, suppress=True, linewidth=120)

    robomaster.config.ROBOT_IP_STR = "192.168.50.113"
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="sta", sn="3JKCH8800100YN")
    ep_chassis = ep_robot.chassis
    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=True, resolution=camera.STREAM_360P)

    
    apriltag = AprilTagDetector(camera_matrix, threads=2, marker_size_m=ARUCO_SIZE)
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