from ultralytics import YOLO
import cv2
import time
import numpy as np
from robomaster import robot
from robomaster import camera

def apply_nms(boxes_xyxy, scores, iou_threshold=0.4):
    """Apply Non-Maximum Suppression to remove duplicate detections."""
    if len(boxes_xyxy) == 0:
        return []
    
    x1 = boxes_xyxy[:, 0]
    y1 = boxes_xyxy[:, 1]
    x2 = boxes_xyxy[:, 2]
    y2 = boxes_xyxy[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]  # Sort by confidence score descending
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        # Compute IoU of the kept box with the rest
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        # Keep boxes with IoU below threshold
        order = order[1:][iou < iou_threshold]
    
    return keep

print('model')

model = YOLO("better.pt")
print("Initializing Robot...")
ep_robot = robot.Robot()
try:
    ep_robot.initialize(conn_type="sta", sn="3JKCH8800100YN")
except Exception as e:
    print(f"Connection failed: {e}")
    exit()
ep_camera = ep_robot.camera
ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)

while True:
    try:
        frame = ep_camera.read_cv2_image(strategy="newest", timeout=1)
    except Exception:
        continue

    if frame is None:
        continue

    start = time.time()
    if model.predictor:
        model.predictor.args.verbose = False
    result = model.predict(source=frame, show=False, conf=0.7)[0]

    boxes = result.boxes
    if len(boxes) > 0:
        xyxy_array = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()

        # Apply NMS
        keep_indices = apply_nms(xyxy_array, scores, iou_threshold=0.4)

        for i in keep_indices:
            xyxy = xyxy_array[i].flatten()
            cv2.rectangle(frame,
                (int(xyxy[0]), int(xyxy[1])),
                (int(xyxy[2]), int(xyxy[3])),
                color=(0, 0, 255), thickness=2)
            cv2.putText(frame, f"Cone {scores[i]:.2f}",
                (int(xyxy[0]), int(xyxy[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    end = time.time()
    print(f"FPS: {1.0 / (end - start):.1f}")