import cv2
import time
import numpy as np
import traceback
import sys
import tty
import termios
import threading
from queue import Empty
from ultralytics import YOLO
from robomaster import robot, camera
import pupil_apriltags

# ───────────────────────── CONSTANTS ─────────────────────────
MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "best.pt"
ARUCO_SIZE  = 0.16
DRIVE_SPEED = 0.3
TURN_SPEED  = 30
camera_matrix = np.array([[314, 0, 320], [0, 314, 180], [0, 0, 1]])



def flush_camera(robot_obj):
    camera = robot_obj.camera
    chassis = robot_obj.chassis
    chassis.drive_speed(x=0, y=0, z=0)
    print(">>> FLUSHING CAMERA BUFFER...")
    time.sleep(0.5) 
    for _ in range(10):
        camera.read_cv2_image(strategy="newest")
# ───────────────────────── APRIL TAG ─────────────────────────
class AprilTagDetector:
    def __init__(self, K, family="tag36h11", threads=2, marker_size_m=0.16):
        self.camera_params = [K[0,0], K[1,1], K[0,2], K[1,2]]
        self.marker_size_m = marker_size_m
        self.detector = pupil_apriltags.Detector(family, threads)

    def find_tags(self, frame_gray):
        return self.detector.detect(
            frame_gray, estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=self.marker_size_m
        )

def draw_apriltags(frame, detections):
    for det in detections:
        if det.decision_margin < 25:
            continue
        pts = det.corners.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
        tl = tuple(pts[0][0])
        tr = tuple(pts[1][0])
        br = tuple(pts[2][0])
        bl = tuple(pts[3][0])
        cv2.line(frame, tl, br, (0, 0, 255), 2)
        cv2.line(frame, tr, bl, (0, 0, 255), 2)

        # Draw tag ID and distance
        if det.pose_t is not None:
            dist = np.linalg.norm(det.pose_t)
            label = f"ID:{det.tag_id} {dist:.2f}m"
        else:
            label = f"ID:{det.tag_id}"

        cv2.putText(frame, label, (tl[0], tl[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# ───────────────────────── YOLO DRAW ─────────────────────────
def draw_yolo(frame, results):
    if len(results) == 0:
        return frame
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return frame

    xyxy_array  = boxes.xyxy.cpu().numpy()
    cls_array   = boxes.cls.cpu().numpy()
    conf_array  = boxes.conf.cpu().numpy()
    names       = results[0].names  # class id -> name

    for i, box in enumerate(xyxy_array):
        x1, y1, x2, y2 = box.astype(int)
        label = f"{names[int(cls_array[i])]} {conf_array[i]:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return frame

# ───────────────────────── KEYBOARD ──────────────────────────
# Shared velocity state updated by keyboard thread
_vel = {"x": 0.0, "y": 0.0, "z": 0.0}
_running = True

KEYMAP = {
    'w': ( DRIVE_SPEED,  0,           0),
    's': (-DRIVE_SPEED,  0,           0),
    'a': ( 0,            DRIVE_SPEED, 0),
    'd': ( 0,           -DRIVE_SPEED, 0),
    'q': ( 0,            0,           TURN_SPEED),
    'e': ( 0,            0,          -TURN_SPEED),
}

def _getch():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

def keyboard_thread():
    global _running
    print("\n[Teleop] Controls: W/S=fwd/back  A/D=strafe  Q/E=turn  SPACE=stop  F=flush camera  CTRL+C=quit\n")
    while _running:
        ch = _getch().lower()
        if ch in KEYMAP:
            _vel["x"], _vel["y"], _vel["z"] = KEYMAP[ch]
        elif ch == ' ':
            _vel["x"] = _vel["y"] = _vel["z"] = 0.0
            print("[Teleop] STOP")
        elif ch == 'f':
            print("[Teleop] Flushing camera...")
            flush_camera(ep_robot)
            print("[Teleop] Camera flushed.")
        elif ch == '\x03':   # CTRL+C
            _running = False
            break

# ───────────────────────── MAIN ──────────────────────────────
if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)

    print(f"[Teleop] Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print("[Teleop] Connecting to robot...")
    robomaster.config.ROBOT_IP_STR = "192.168.50.113"
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="sta", sn="3JKCH8800100YN")
    ep_chassis = ep_robot.chassis
    ep_camera  = ep_robot.camera
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)

    apriltag = AprilTagDetector(camera_matrix, marker_size_m=ARUCO_SIZE)

    # Start keyboard input thread (non-blocking)
    kb_thread = threading.Thread(target=keyboard_thread, daemon=True)
    kb_thread.start()

    try:
        while _running:
            # ── Read frame ──
            try:
                frame = ep_camera.read_cv2_image(strategy="newest", timeout=1)
            except Empty:
                continue
            if frame is None:
                continue

            # ── YOLO detection ──
            results = model(frame, conf=0.5, verbose=False)
            frame = draw_yolo(frame, results)

            # ── AprilTag detection ──
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.uint8)
            detections = apriltag.find_tags(gray)
            draw_apriltags(frame, detections)

            # ── HUD: velocity overlay ──
            hud = f"x:{_vel['x']:+.2f}  y:{_vel['y']:+.2f}  z:{_vel['z']:+.1f}"
            cv2.putText(frame, hud, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "W/S:fwd  A/D:strafe  Q/E:turn  SPACE:stop",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            cv2.imshow("Teleop View", frame)
            if cv2.waitKey(1) & 0xFF == ord('`'):
                break

            # ── Send velocity to robot ──
            ep_chassis.drive_speed(x=_vel["x"], y=_vel["y"], z=_vel["z"])

    except KeyboardInterrupt:
        pass
    except Exception:
        print(traceback.format_exc())
    finally:
        _running = False
        print("[Teleop] Shutting down...")
        ep_chassis.drive_speed(x=0, y=0, z=0)
        time.sleep(0.5)
        ep_camera.stop_video_stream()
        cv2.destroyAllWindows()
        ep_robot.close()