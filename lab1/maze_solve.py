import pupil_apriltags
import cv2
import numpy as np
import time
import traceback
from queue import Empty
import robomaster
from robomaster import robot
from robomaster import camera
#import map
import apriltag

BOX_SIZE_MM = 266
BOX_SIZE_M = BOX_SIZE_MM/1000

# source ~/py38-venv/bin/activate

if __name__ == '__main__':

    robomaster.config.ROBOT_IP_STR = "192.168.50.113"
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="sta",sn="3JKCH8800100YN")

    ep_chassis = ep_robot.chassis

    ep_chassis.move(x=0, y=0, z=90, xy_speed=1).wait_for_completed()
