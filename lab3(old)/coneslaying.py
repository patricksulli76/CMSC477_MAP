import cv2
import time
import numpy as np
from ultralytics import YOLO
from robomaster import robot

if __name__ == '__main__':
    MODEL_PATH = "Cone.pt"
    
    print("Initializing Robot...")
    ep_robot = robot.Robot()
    try:
        ep_robot.initialize(conn_type="sta", sn="3JKCH8800100YN")
    except Exception as e:
        print(f"Connection failed: {e}")
        exit()

    time.sleep(3)
    print("Connection Successful...")

    ep_chassis = ep_robot.chassis
    ep_arm = ep_robot.robotic_arm
    ep_gripper = ep_robot.gripper
    ep_camera = ep_robot.camera
    ep_camera.start_video_stream()  # Start the camera stream
    model = YOLO(MODEL_PATH)

    try:
        while True:
            try:
                frame = ep_camera.read_cv2_image(strategy="newest", timeout=1)
            except Exception:
                # Camera timeout or read error, skip this frame silently
                continue
            
            if frame is None:
                continue

            start = time.time()

            results = model(frame, conf=0.4, verbose=False)
            
            # Draw boxes on frame
            annotated_frame = frame.copy()
            
            if len(results) > 0:
                result = results[0]
                
                # Extract boxes
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes.xyxy  # Get tensor
                    
                    # Convert tensor to numpy
                    if hasattr(boxes, 'cpu'):
                        xyxy_array = boxes.cpu().numpy()
                    else:
                        xyxy_array = np.array(boxes)
                    
                    # Draw rectangles
                    for box in xyxy_array:
                        x1, y1, x2, y2 = box.astype(int)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(annotated_frame, "Cone", (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.imshow('Cone Detection', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            end = time.time()
            elapsed = end - start
            if elapsed > 0:
                print(f"FPS: {1.0 / elapsed:.1f}")
                
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ep_camera.stop_video_stream()
        cv2.destroyAllWindows()
        try:
            ep_robot.close()
        except Exception:
            pass