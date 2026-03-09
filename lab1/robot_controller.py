import robomaster
from robomaster import robot


class Robot:
    def __init__(self):
        robomaster.config.ROBOT_IP_STR = "192.168.50.113"
        self.ep_robot = robot.Robot()
        self.ep_robot.initialize(conn_type="sta",sn="3JKCH8800100YN")
        self.speed = 0.5
        self.ep_chassis = self.ep_robot.chassis


    def move_to(self,ep_chassis,curr_pos,target_pos):
        goal_x,goal_y = (target_pos[0]-curr_pos[0],target_pos[1]-curr_pos[1])

        if(abs(goal_x-curr_pos[0]) < .1):
            goal_x = 0
        if(abs(goal_y-curr_pos[1]) < .1):
            goal_y = 0

        
        if (goal_y > 0):
            print("moving Y")
            self.ep_chassis.move(x=0, y=-goal_y*.9, z=0, xy_speed=0.2).wait_for_completed()
        elif (goal_y < 0):
            print("moving Y")
            self.ep_chassis.move(x=0, y=goal_y*.9, z=0, xy_speed=0.2).wait_for_completed()
        if (goal_x > 0):
            print("moving X")
            self.ep_chassis.move(x=goal_x, y=0, z=0, xy_speed=0.2).wait_for_completed()
        elif (goal_x < 0):
            print("moving X")
            self.ep_chassis.move(x=-goal_x, y=0, z=0, xy_speed=0.2).wait_for_completed()
