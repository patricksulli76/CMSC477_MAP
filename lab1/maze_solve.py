import pupil_apriltags
import cv2
import numpy as np
import time
import traceback
from queue import Empty
from robomaster import robot
from robomaster import camera
import map
import apriltag

BOX_SIZE_MM = 266



np.set_printoptions(precision=3, suppress=True, linewidth=120)

ep_robot = robot.Robot()
ep_robot.initialize(conn_type="ap")#(conn_type="sta", sn="3JKCH7T00100J0")
ep_chassis = ep_robot.chassis
ep_camera = ep_robot.camera
ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)

K = np.array([[314, 0, 320], [0, 314, 180], [0, 0, 1]]) # Camera focal length and center pixel
marker_size_m = 0.153 # Size of the AprilTag in meters
detector = AprilTagDetector(K, threads=2, marker_size_m=marker_size_m)

try:
    detect_tag_loop(ep_robot, ep_chassis, ep_camera, detector)
except KeyboardInterrupt:
    pass
except Exception as e:
    print(traceback.format_exc())
finally:
    print('Waiting for robomaster shutdown')
    ep_camera.stop_video_stream()
    ep_robot.close()