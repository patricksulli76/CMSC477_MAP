# Needs to be buried into train4 weights where best.pt is hiding
import cv2
import time
import numpy as np
from ultralytics import YOLO
from robomaster import robot

def execute_pick_sequence(robot_obj):
    arm = robot_obj.robotic_arm
    gripper = robot_obj.gripper
    print(">>> EXECUTING GROUND-LEVEL GRAB")
    arm.moveto(x=185, y=-95).wait_for_completed() 
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
    chassis.move(z=-90).wait_for_completed()
    
    print(">>> MOVING 0.5M TO DELIVERY POINT")
    chassis.move(x=0.5).wait_for_completed()
    
    print(">>> LOWERING AND RELEASING")
    arm.moveto(x=185, y=-95).wait_for_completed()
    gripper.open(power=50)
    time.sleep(1.0)

    print(">>> RETURNING TO SEARCH AREA")
    chassis.move(x=-1).wait_for_completed() 
    arm.moveto(x=180, y=-90).wait_for_completed() 
    chassis.move(z=90).wait_for_completed()
    chassis.move(x=0.3).wait_for_completed() 

    print(">>> FLUSHING CAMERA BUFFER...")
    time.sleep(1.5) 
    for _ in range(10):
        camera.read_cv2_image(strategy="newest")
    
    print(">>> READY FOR NEXT TARGET")

if __name__ == '__main__':
    MODEL_PATH = "best.pt"
    CENTER_X = 320         
    VISIBLE_GRAB_THRESHOLD = 240 
    BLIND_SPOT_Y = 410     
    DRIVE_SPEED = 0.08     
    SEARCH_TIMEOUT = 15.0 

    print("Initializing Robot...")
    ep_robot = robot.Robot()
    try:
        ep_robot.initialize(conn_type="sta", sn="3JKCH8800100YN")
    except Exception as e:
        print(f"Connection failed: {e}")
        exit()

    ep_chassis = ep_robot.chassis
    ep_arm = ep_robot.robotic_arm
    ep_gripper = ep_robot.gripper
    ep_camera = ep_robot.camera
    model = YOLO(MODEL_PATH)

    ep_arm.moveto(x=180, y=-90).wait_for_completed()
    ep_gripper.open(power=50)
    time.sleep(1)   

    ep_camera.start_video_stream(display=False, resolution="360p")
    time.sleep(2) 

    last_y = 0
    last_x = 0
    last_seen_time = time.time()

    try:
        print(f"Continuous Loop Started. Timeout: {SEARCH_TIMEOUT}s.")
        while True:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=2)
            if img is None: continue

            results = model(img, conf=0.5, verbose=False)
            
            if len(results[0].boxes) > 0:
                # --- LEGO FOUND: LOCK ON ---
                last_seen_time = time.time() 
                box = results[0].boxes[0]
                coords = box.xyxy[0].cpu().numpy()
                last_x = (coords[0] + coords[2]) / 2
                last_y = coords[3] 
                
                error_x = CENTER_X - last_x
                
                if abs(error_x) > 40:
                    ep_chassis.drive_speed(x=0, y=error_x * -0.002, z=error_x * 0.008)
                elif last_y > VISIBLE_GRAB_THRESHOLD:
                    ep_chassis.drive_speed(x=0, y=0, z=0)
                    execute_pick_sequence(ep_robot)
                    execute_delivery_sequence(ep_robot)
                    last_y, last_x = 0, 0
                    last_seen_time = time.time()
                else:
                    ep_chassis.drive_speed(x=DRIVE_SPEED, y=0, z=error_x * 0.005)
            
            else:
                # --- LEGO LOST ---
                if last_y > BLIND_SPOT_Y and abs(last_x - CENTER_X) < 60:
                    # Blind spot logic remains the same
                    last_seen_time = time.time() 
                    ep_chassis.drive_speed(x=0, y=0, z=0)
                    ep_chassis.move(x=0.04).wait_for_completed()
                    execute_pick_sequence(ep_robot)
                    execute_delivery_sequence(ep_robot)
                    last_y, last_x = 0, 0
                    last_seen_time = time.time()
                else:
                    # TRULY LOST: START 360 SCAN
                    time_since_last_seen = time.time() - last_seen_time
                    if time_since_last_seen > SEARCH_TIMEOUT:
                        print(f">>> AREA CLEAR AFTER SCAN. MISSION END.")
                        break 
                    
                    # Spin at 20 deg/s to find next target
                    # If it sees a LEGO, the 'if len(boxes) > 0' block will interrupt this
                    ep_chassis.drive_speed(x=0, y=0, z=20) 

            annotated_frame = results[0].plot()
            cv2.imshow("RoboMaster YOLO View", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            time.sleep(0.01)

    except Exception as e:
        print(f"System Error: {e}")
    finally:
        ep_chassis.drive_speed(x=0, y=0, z=0)
        cv2.destroyAllWindows()
        ep_camera.stop_video_stream()
        ep_robot.close()
