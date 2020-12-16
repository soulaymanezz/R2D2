import argparse
import cv2
import pyrealsense2 as rs
import time
import numpy as np

from yolo import YOLO



ap = argparse.ArgumentParser()
ap.add_argument('-n', '--network', default="normal", help='Network Type: normal / tiny / prn')
ap.add_argument('-d', '--device', default=0, help='Device to use')
ap.add_argument('-s', '--size', default=416, help='Size for yolo')
ap.add_argument('-c', '--confidence', default=0.2, help='Confidence for yolo')
args = ap.parse_args()

if args.network == "normal":
    print("loading yolo...")
    yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
elif args.network == "prn":
    print("loading yolo-tiny-prn...")
    yolo = YOLO("models/cross-hands-tiny-prn.cfg", "models/cross-hands-tiny-prn.weights", ["hand"])
else:
    print("loading yolo-tiny...")
    yolo = YOLO("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])



# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

#pipeline.start()
yolo.size = int(args.size)
yolo.confidence = float(args.confidence)

print("starting webcam...")
def distance(frames):
    # Wait for a coherent pair of frames: depth and color
    
    depth_frame = frames.get_depth_frame()
    frame = frames.get_color_frame()
    frame = np.asanyarray(frame.get_data())
    

    frame_dpt = depth_frame.as_depth_frame()
    #frame_dpt = pipeline.wait_for_frames().get_depth_frame().as_depth_frame()
    width, height, inference_time, results = yolo.inference(frame)
    for detection in results:
        id, name, confidence, x, y, w, h = detection
        cx = int(x + (w / 2))
        cy = int(y + (h / 2))
        print((cx,cy))
        distance_to_hand = frame_dpt.get_distance(cx, cy)
    try:
        return distance_to_hand
    except:
        return "ba9i"
        # draw a bounding box rectangle and label on the image
        #color = (224, 41, 9)
        #cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        #text = "%s (%s)" % (name, round(distance_to_hand,2))
        #cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)


while True:
    frames = pipeline.wait_for_frames()
    DISTANCE = distance(frames)
    if DISTANCE != "ba9i":
        break
    


