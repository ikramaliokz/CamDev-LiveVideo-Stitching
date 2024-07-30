import cv2
import numpy as np
import math
from config import NN_SIZE, bbox_color, text_color, constant, min_radius, max_radius, labelMap, METER_TO_PIXEL
import time

class TextHelper:
    def __init__(self, color = (255, 255, 255), text_color = (255, 255, 255)) -> None:
        self.bg_color = (0, 0, 0)
        self.color = color
        self.text_color = text_color
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 0.5, self.bg_color, 3, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 0.5, self.text_color, 1, self.line_type)
    def rectangle(self, frame, bbox):
        x1,y1,x2,y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.bg_color, 3)
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.color, 1)

class FPSHandler:
    def __init__(self):
        self.timestamp = time.time() + 1
        self.start = time.time()
        self.frame_cnt = 0
    def next_iter(self):
        self.timestamp = time.time()
        self.frame_cnt += 1
    def fps(self):
        return self.frame_cnt / (self.timestamp - self.start)

def frameNorm(frame, NN_SIZE, bbox):
    # Check difference in aspect ratio and apply correction to BBs
    ar_diff = NN_SIZE[0] / NN_SIZE[0] - frame.shape[0] / frame.shape[1]
    sel = 0 if 0 < ar_diff else 1
    bbox[sel::2] *= 1-abs(ar_diff)
    bbox[sel::2] += abs(ar_diff)/2
    # Normalize bounding boxes0
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(bbox, 0, 1) * normVals).astype(int)

def displayFrame(name, detections, frame, ani_frame, text, radius1, radius2, radius3):
    for detection in detections:
        
        bbox = frameNorm(frame, NN_SIZE, np.array([detection.xmin, detection.ymin, detection.xmax, detection.ymax]))
        text.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20))
        text.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40))
        text.putText(frame, f"Z: {int(detection.spatialCoordinates.z)/1000} m", (bbox[0] + 10, bbox[1] + 60))
        text.putText(frame, f"X: {int(detection.spatialCoordinates.x)/1000} m", (bbox[0] + 10, bbox[1] + 80))
        text.putText(frame, f"Y: {int(detection.spatialCoordinates.y)/1000} m", (bbox[0] + 10, bbox[1] + 100))

        text.rectangle(frame, bbox)
        draw_animation(frame, radius1, radius2, radius3, int(detection.spatialCoordinates.x)/1000,int(detection.spatialCoordinates.z)/1000, METER_TO_PIXEL)
        draw_animation(ani_frame,radius1, radius2, radius3, int(detection.spatialCoordinates.x)/1000,int(detection.spatialCoordinates.z)/1000, METER_TO_PIXEL)

    # Show the frame
    cv2.imshow(name, frame)

def draw_animation(frame, radius1, radius2, radius3,  x_point, z_point, m_to_pix):
    # Get frame dimensions
    height, width, _ = frame.shape
    x_point_pixel = math.trunc(x_point*m_to_pix)
    z_point_pixel = math.trunc(z_point * m_to_pix)
    # Calculate the center point of the bounding box

    # Define the starting point (mid bottom of the frame)
    start_point = (width // 2, height)
    end_point = (start_point[0] + x_point_pixel, start_point[1] - z_point_pixel)

    # print(start_point, end_point)
    if z_point_pixel < radius1:
        cv2.circle(frame, start_point, radius1, (255, 255, 255), 2)
        cv2.circle(frame, start_point, radius2, (255, 255, 255), 2)
        cv2.circle(frame, start_point, radius3, (255, 255, 255), 2)

    elif z_point_pixel < radius2:
        cv2.circle(frame, start_point, radius2, (255, 255, 255), 2)
        cv2.circle(frame, start_point, radius3, (255, 255, 255), 2)

    elif z_point_pixel < radius3:
        cv2.circle(frame, start_point, radius3, (255, 255, 255), 2)

    # Draw the line on the frame
    cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
    cv2.circle(frame, end_point,5, (255, 255, 255), 5, -1)
    text = f"X (meters): {x_point}, Z(meters): {z_point}"
    cv2.putText(frame, text, (end_point[0]+ 10, end_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3, cv2.LINE_AA)
    
def calculate_radius(speed):
    radius = constant / np.log(speed + 1)  # Add 1 to avoid log(0)
    return int(max(min(radius, max_radius), min_radius))


