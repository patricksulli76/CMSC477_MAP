# -*-coding:utf-8-*-
# Copyright (c) 2020 DJI.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License in the file LICENSE.txt or at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import robomaster
from robomaster import robot, camera
import cv2
import time
import os

if __name__ == '__main__':
    # Configuration
    robomaster.config.ROBOT_IP_STR = "192.168.50.113"
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="sta", sn="3JKCH8800100YN")
    time.sleep(2)
    ep_camera = ep_robot.camera

    # Start video stream without the built-in display 
    # so we can manage the window ourselves with OpenCV
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)

    print("Live feed started.")
    print("Commands: \n  Press 's' to Save Image \n  Press 'q' to Quit")

    img_count = 0
    folder_name = "images"
    os.makedirs(folder_name, exist_ok=True)  # Create folder if it doesn't exist

    try:
        while True:
            # Get the latest frame from the robot
            frame = ep_camera.read_cv2_image(strategy="newest")

            # Display the frame in an OpenCV window
            cv2.imshow("RoboMaster Live Feed", frame)

            # Wait for 1ms for a key press
            key = cv2.waitKey(1) & 0xFF

            # Capture image when 's' is pressed
            if key == ord('s'):
                filename = f"{folder_name}/robomaster_cap_{img_count}.jpg"
                success = cv2.imwrite(filename, frame)
                if success:
                    print(f"Saved: {filename}")
                else:
                    print(f"Failed to save: {filename}")
                img_count += 1

            # Exit loop when 'q' is pressed
            elif key == ord('q'):
                print("Closing stream...")
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    # Cleanup
    cv2.destroyAllWindows()
    ep_camera.stop_video_stream()
    ep_robot.close()
