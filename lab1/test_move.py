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

if __name__ == '__main__':
    # More legible printing from numpy.
    np.set_printoptions(precision=3, suppress=True, linewidth=120)

    robomaster.config.ROBOT_IP_STR = "192.168.50.113"
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="sta", sn="3JKCH8800100YN")
    ep_chassis = ep_robot.chassis
    try:
        ep_chassis.move(x=0, y=0, z=90, z_speed=45).wait_for_completed()
        ep_chassis.move(x=0.2, y=0, z=0, xy_speed=0.2).wait_for_completed()
        ep_chassis.move(x=0.2, y=0, z=0, xy_speed=0.2).wait_for_completed()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(traceback.format_exc())
    finally:
        print('Waiting for robomaster shutdown')
        ep_chassis.drive_speed(x=0, y=0, z=0, timeout=1)
        time.sleep(1)
        ep_robot.close()
