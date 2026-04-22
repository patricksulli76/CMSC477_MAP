# Needs to be buried into train4 weights where best.pt is hiding
import cv2
import time
import numpy as np
from ultralytics import YOLO
from robomaster import robot

####################################

BLOCK_SIZE = (0.03, 0.175) # w,h

camera_matrix = np.array([[314, 0, 320], [0, 314, 180], [0, 0, 1]])

def solve_pnp_rectangle(corners_2d, dist_coeffs=None):
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5)

    half_w = BLOCK_SIZE[0] / 2.0
    half_h = BLOCK_SIZE[1] / 2.0

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


DEFAULTS = {
    "Red":   {"HL": 165, "HH": 179, "SL": 50, "SH": 255, "VL": 150, "VH": 255},
    "Green": {"HL": 35, "HH": 85, "SL": 80, "SH": 255, "VL": 60, "VH": 255},
}

RED_RANGE   = {"lower": np.array([150, 50, 150]), "upper": np.array([179, 255, 255])}
GREEN_RANGE = {"lower": np.array([35, 80, 60]),   "upper": np.array([85, 255, 255])}

BOX_COLORS = {
    "Red":   (0, 0, 255),   # red in BGR
    "Green": (0, 255, 0),   # green in BGR
}



MIN_AREA = 200
APPROX_EPS = 0.04

def _order_corners(approx):
    """
    Order 4 polygon vertices as: top-left, top-right, bottom-right, bottom-left.
    This matches the obj_pts convention in solve_pnp_rectangle:
        TL (-w, +h)  TR (+w, +h)  BR (+w, -h)  BL (-w, -h)
    """
    pts = approx.reshape(4, 2).astype(np.float64)
    # sort by y first: top two vs bottom two
    s = pts.sum(axis=1)       # x+y  → smallest = TL, largest = BR
    d = np.diff(pts, axis=1).flatten()  # y-x → smallest = TR, largest = BL
    ordered = np.zeros((4, 2), dtype=np.float64)
    ordered[0] = pts[np.argmin(s)]   # TL
    ordered[1] = pts[np.argmin(d)]   # TR
    ordered[2] = pts[np.argmax(s)]   # BR
    ordered[3] = pts[np.argmax(d)]   # BL
    return ordered


def build_mask(hsv, lower, upper):
    """Threshold + morphological cleanup."""
    mask = cv2.inRange(hsv, lower, upper)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    return mask


def localize_rectangles(image, camera_matrix, rect_width, rect_height,
                        dist_coeffs=None, min_area=MIN_AREA):
    """
    Detect the largest red and green rectangles in an image and return their
    translation vectors (pose) relative to the camera.
 
    Parameters
    ----------
    image : np.ndarray
        BGR image (e.g. from cv2.imread or cap.read()).
    camera_matrix : np.ndarray (3,3)
        Camera intrinsic matrix.
    rect_width : float
        Physical width of the rectangle (units carry to tvec output).
    rect_height : float
        Physical height of the rectangle.
    dist_coeffs : np.ndarray, optional
        Distortion coefficients. Defaults to zero distortion.
    min_area : int
        Ignore contours smaller than this (px²).
 
    Returns
    -------
    (poses, annotated_image)
        poses : {"red":   {"tvec": np.ndarray (3,1), "rvec": np.ndarray (3,1),
                           "corners": np.ndarray (4,2)} or None,
                 "green": ... or None}
        annotated_image : np.ndarray – copy of input with contours, corners,
                          and tvec drawn
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5)
 
    half_w = rect_width / 2.0
    half_h = rect_height / 2.0
    obj_pts = np.array([
        [-half_w,  half_h, 0],
        [ half_w,  half_h, 0],
        [ half_w, -half_h, 0],
        [-half_w, -half_h, 0],
    ], dtype=np.float64)
 
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    annotated = image.copy()
 
    draw_colors = {"red": (0, 0, 255), "green": (0, 255, 0)}
    corner_labels = ["TL", "TR", "BR", "BL"]
    result = {"red": None, "green": None}
 
    for key, rng in [("red", RED_RANGE), ("green", GREEN_RANGE)]:
        mask = build_mask(hsv, rng["lower"], rng["upper"])
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
 
        # pick the largest contour above min_area (any shape)
        valid = [(cv2.contourArea(c), c) for c in contours
                 if cv2.contourArea(c) >= min_area]
 
        if not valid:
            continue
 
        _, biggest = max(valid, key=lambda x: x[0])
 
        # fit a rotated rectangle to whatever shape was found
        rect = cv2.minAreaRect(biggest)          # ((cx,cy), (w,h), angle)
        box = cv2.boxPoints(rect).astype(np.float64)  # 4 corners
        corners = _order_corners(box)
 
        # ── solvePnP ──
        img_pts = corners.astype(np.float64).reshape(4, 1, 2)
        success, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE
        )
        if not success:
            continue
 
        result[key] = {"tvec": tvec, "rvec": rvec, "corners": corners}
 
        # ── draw ──
        color = draw_colors[key]
        cv2.drawContours(annotated, [biggest], -1, color, 3)
        pts_int = corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts_int], True, (255, 255, 255), 2)
 
        for i, (cx, cy) in enumerate(corners.astype(int)):
            cv2.circle(annotated, (cx, cy), 6, color, -1)
            cv2.circle(annotated, (cx, cy), 6, (255, 255, 255), 2)
            cv2.putText(annotated, corner_labels[i], (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
 
        # draw axes at rectangle center
        cv2.drawFrameAxes(annotated, camera_matrix, dist_coeffs,
                          rvec, tvec, rect_width * 0.4)
 
        # tvec label at centroid
        M = cv2.moments(biggest)
        if M["m00"] != 0:
            mcx = int(M["m10"] / M["m00"])
            mcy = int(M["m01"] / M["m00"])
            tx, ty, tz = tvec.flatten()
            txt = f"{key.capitalize()} t=[{tx:.3f}, {ty:.3f}, {tz:.3f}]"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (mcx - tw // 2 - 2, mcy - th // 2 - 4),
                           (mcx + tw // 2 + 2, mcy + th // 2 + 4), color, -1)
            cv2.putText(annotated, txt, (mcx - tw // 2, mcy + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
 
    return result, annotated



 ####################################
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

def execute_delivery_sequence(robot_obj):
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

def navigate_to_zone(robot_obj,zone_color,frame):
    corners = None
    poses, annotated = localize_rectangles(
    frame,
    camera_matrix,
    rect_width=0.216,    # physical width in metres
    rect_height=0.279,   # physical height in metres
    )

    if poses[zone_color] is not None:
        tvec = poses[zone_color]["tvec"]   # (3,1) translation
        rvec = poses[zone_color]["rvec"]   # (3,1) Rodrigues rotation
        corners = poses[zone_color]["corners"]  # (4,2) image corners TL/TR/BR/BL
    return annotated,poses[zone_color]
    

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

if __name__ == '__main__':
    MODEL_PATH = "best.pt"
    #MODEL_PATH = "better.pt"
    CENTER_X = 320    
    CENTER_Y = 180           
    VISIBLE_GRAB_THRESHOLD = 320
    BLIND_SPOT_Y = 410     
    DRIVE_SPEED = 0.04     
    SEARCH_TIMEOUT = 15.0 

    print("Initializing Robot...")
    ep_robot = robot.Robot()
    try:
        ep_robot.initialize(conn_type="sta", sn="3JKCH8800100YN")
    except Exception as e:
        print(f"Connection failed: {e}")
        exit()

    time.sleep(3)

    ep_chassis = ep_robot.chassis
    ep_arm = ep_robot.robotic_arm
    ep_gripper = ep_robot.gripper
    ep_camera = ep_robot.camera
    model = YOLO(MODEL_PATH)

    ep_arm.moveto(x=180, y=-10).wait_for_completed()
    ep_gripper.open(power=50)
    time.sleep(1)   

    ep_camera.start_video_stream(display=False, resolution="360p")
    time.sleep(2) 

    last_y = 0
    last_x = 0
    last_seen_time = time.time()
    last_flush = time.time()

    states = {
        0: "SEARCHING",
        1: "LOCKED IN",
        2: "PICKUP",
        3: "DELIVERING"
    }

    real_states = {
        0: "SEARCHING ZONE",
        1: "SEARCHING BLOCK",
        2: "DELIVERING TO TEMP ZONE",
        3: "DELIVERING TO DROP ZONE",
        4: "END MISSION",

    }

    curr_state = 0
    holding_block = False
    curr_color = "green"
    got_temp = False
    lr_counter = 0

    try:
        print(f"Continuous Loop Started. Timeout: {SEARCH_TIMEOUT}s.")
        while True:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=2)
            if img is None: continue
            annotated_frame = None
            results = model(img, conf=0.5, verbose=False)

            # ------------------------ STATE MACHINE------------------------
            if(time.time() - last_flush) > 8:
                flush_camera(ep_robot)
                ep_chassis.drive_speed(x=0, y=0, z=0)
                last_flush = time.time()


            # ------------------------ STATE 0: SEARCHING ZONE -------------
            if curr_state == 0:
                annotated_frame, pose = navigate_to_zone(ep_robot, curr_color, img)

                if pose is not None:
                    Z = Z = pose["tvec"][2][0]  # depth in meters
                    corners_2d = pose["corners"]  # (4,2) image corners TL/TR/BR/BL

                    center_X_m, center_Y_m = pixel_to_camera_coords(CENTER_X, CENTER_Y, Z, camera_matrix)
                    center_x, center_y = corners_2d.mean(axis=0)

                    print("Center x: ",center_x)
                    print("Center x_m: ",center_X_m)
                    print("Z: ",Z)


                    if ((center_x < CENTER_X - 20) and holding_block == False) or ((center_x < CENTER_X - 80 and holding_block == True)):
                        print("Moving Left")
                        lr_counter +=1
                        ep_chassis.drive_speed(x=0, y=0, z=-5)
                    elif ((center_x > CENTER_X + 20) and holding_block == False) or ((center_x > CENTER_X + 80 and holding_block == True)):
                        print("Moving Right")
                        lr_counter +=1
                        ep_chassis.drive_speed(x=0, y=0, z=5)
                    else:
                        print(center_x, CENTER_X)
                        print("Moving Forward")
                        ep_chassis.drive_speed(x=0.2, y=0, z=0)

                    print("LR Counter: ", lr_counter)
                    if (Z < 0.4 and holding_block == False) or (lr_counter > 150 and holding_block == True):
                        ep_arm.moveto(x=180, y=-10).wait_for_completed()
                        ep_gripper.open(power=50)
                        time.sleep(1)   
                        curr_state = 1
                        lr_counter = 0
                        if holding_block == True:
                            ep_chassis.move(x=-0.2).wait_for_completed()
                            ep_chassis.move(z=-90).wait_for_completed()
                            holding_block = False
                        flush_camera(ep_robot)
                        last_flush = time.time()

                            

                else:
                    ep_chassis.drive_speed(x=0, y=0, z=10)

                    
                cv2.imshow("RoboMaster YOLO View", annotated_frame)

            # ------------------------ STATE 0: SEARCHING BLOCK ----------

            if curr_state != 1:
                last_read_depths = [100,100,100]
                curr_depth_index = 0
            if curr_state == 1:
                # Show YOLO detection results for blocks
                annotated_frame = results[0].plot()



                if len(results[0].boxes) > 0:
                    last_seen_time = time.time()
                    box = results[0].boxes[0]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    # Order: TL, TR, BR, BL
                    corners_2d = np.array([
                        [x1, y1],
                        [x2, y1],
                        [x2, y2],
                        [x1, y2]
                    ], dtype=np.float64)
                    tvec = solve_pnp_rectangle(corners_2d)
                    x_dist = tvec[0][0]  # Lateral (Left/Right)
                    y_dist = tvec[2][0]  # Depth (Forward/Distance)
                    print(f"Block X (Lateral): {x_dist:.2f}")
                    print(f"Block Y (Depth): {y_dist:.2f}")

                    last_read_depths[curr_depth_index] = y_dist
                    curr_depth_index += 1

                    if curr_depth_index > 2:
                        curr_depth_index = 0

                    if(x_dist < -0.02):
                        print("Block to the Left, moving Left")
                        ep_chassis.drive_speed(x=0, y=0, z=-5)
                    elif(x_dist > 0.02):
                        print("Block to the Right, moving Right")
                        ep_chassis.drive_speed(x=0, y=0, z=5)
                    else:
                        ep_chassis.drive_speed(x=0.12, y=0, z=0)

                    if(sum(last_read_depths)/3 < 0.15):
                        print("Block within reach, executing pick se0quence")
                        ep_chassis.drive_speed(x=0, y=0, z=0)
                        execute_pick_sequence(ep_robot)
                        if(got_temp == False):
                            curr_state = 2
                        else:
                            curr_color = "red" if curr_color == "green" else "green"
                            ep_arm.moveto(x=180, y=-20).wait_for_completed()
                            time.sleep(1)   
                            ep_chassis.move(z=180).wait_for_completed()
                            time.sleep(1)
                            holding_block = True
                            lr_counter = 0
                            curr_state = 0
                else:
                    if(time.time() - last_seen_time) > 2.0:
                        ep_chassis.drive_speed(x=0, y=0, z=-5)
                    # Use x_dist, y_dist for navigation logic
                    # Example: move forward if y_dist > threshold, align if x_dist is off-center
            if curr_state == 2:
                print("Delivering to temp zone")
                execute_delivery_sequence(ep_robot)
                curr_color = "red" if curr_color == "green" else "green"
                got_temp = True
                curr_state = 0
                lr_counter = 0
            if annotated_frame is not None:
                cv2.imshow("RoboMaster YOLO View", annotated_frame)
            # if len(results[0].boxes) > 0:
            #     box = results[0].boxes[0]

            #     box = results[0].boxes[0]
            #     # Extract x1, y1, x2, y2 from the tensor
            #     x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            #     # Construct the 4 corners in the SPECIFIC order used in obj_pts:
            #     # 1. Top-Left:     (x1, y1)
            #     # 2. Top-Right:    (x2, y1)
            #     # 3. Bottom-Right: (x2, y2)
            #     # 4. Bottom-Left:  (x1, y2)
            #     corners_2d = np.array([
            #         [x1, y1],
            #         [x2, y1],
            #         [x2, y2],
            #         [x1, y2]
            #     ], dtype=np.float64)
            #     tvec = solve_pnp_rectangle(corners_2d)

            #     x_dist = tvec[0][0]  # Lateral (Left/Right)
            #     y_dist = tvec[2][0]  # Depth (Forward/Distance)

            #     print(f"X (Lateral): {x_dist:.2f}")
            #     print(f"Y (Vertical): {y_dist:.2f}")
                
            #     # Total straight-line distance
            #     distance = np.linalg.norm(tvec)
            #     print(f"Total Distance: {distance:.2f}")


            # annotated_frame = navigate_to_zone(ep_robot, "green", img)
            # #print(len(results[0].boxes))
            # #annotated_frame = results[0].plot()
            # cv2.imshow("RoboMaster YOLO View", annotated_frame)

            


            # if (curr_state == 0 or curr_state == 1) and time.time() - last_flush > 8:
            #     flush_camera(ep_robot)
            #     last_flush = time.time()
            # if len(results[0].boxes) > 0:
            #     # --- LEGO FOUND: LOCK ON ---
            #     curr_state = 1
            #     last_seen_time = time.time() 
            #     box = results[0].boxes[0]
            #     coords = box.xyxy[0].cpu().numpy()
            #     last_x = (coords[0] + coords[2]) / 2
            #     last_y = coords[3] 
                
            #     error_x = CENTER_X - last_x
                
            #     if abs(error_x) > 40:
            #         ep_chassis.drive_speed(x=0, y=error_x * -0.002, z=error_x * 0.008)
            #     elif last_y > VISIBLE_GRAB_THRESHOLD:
            #         ep_chassis.drive_speed(x=0, y=0, z=0)
            #         curr_state = 2
            #         execute_pick_sequence(ep_robot)
            #         curr_state = 3
            #         execute_delivery_sequence(ep_robot)
            #         last_y, last_x = 0, 0
            #         last_seen_time = time.time()
            #     else:
            #         ep_chassis.drive_speed(x=DRIVE_SPEED, y=0, z=error_x * 0.005)
            
            # elif (time.time() - last_seen_time) > 1.0:
            #     # --- LEGO LOST ---
            #     curr_state = 0
            #     if last_y > BLIND_SPOT_Y and abs(last_x - CENTER_X) < 60:
            #         # Blind spot logic remains the same
            #         last_seen_time = time.time() 
            #         ep_chassis.drive_speed(x=0, y=0, z=0)
            #         ep_chassis.move(x=0.04).wait_for_completed()
            #         execute_pick_sequence(ep_robot)
            #         execute_delivery_sequence(ep_robot)
            #         last_y, last_x = 0, 0
            #         last_seen_time = time.time()
            #     else:
            #         # TRULY LOST: START 360 SCAN
            #         time_since_last_seen = time.time() - last_seen_time
            #         if time_since_last_seen > SEARCH_TIMEOUT:
            #             print(f">>> AREA CLEAR AFTER SCAN. MISSION END.")
            #             break 
                    
            #         # Spin at 20 deg/s to find next target
            #         # If it sees a LEGO, the 'if len(boxes) > 0' block will interrupt this
            #         ep_chassis.drive_speed(x=0, y=0, z=5) 

            # annotated_frame = results[0].plot()
            # cv2.imshow("RoboMaster YOLO View", annotated_frame)


            if cv2.waitKey(1) & 0xFF == ord('q'): break
            time.sleep(0.01)

    except Exception as e:
        print(f"System Error: {e}")
    finally:
        ep_chassis.drive_speed(x=0, y=0, z=0)
        cv2.destroyAllWindows()
        ep_camera.stop_video_stream()
        ep_robot.close()
