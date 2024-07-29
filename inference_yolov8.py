import cv2

import depthai as dai
import numpy as np
import time
import blobconverter
import math
from utility import *

NN_SIZE = (640,640)
METER_TO_PIXEL = 50  # value for one meter length in pixels

min_radius = 50
max_radius = 800
constant = 800
text_color = (0, 0, 255)
bbox_color = (0, 255, 0)
nnBlobPath = "models/yolov8n_openvino_2022.1_6shave.blob"

labelMap = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "TV", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Start defining a pipeline
pipeline = dai.Pipeline()

# Create output streams
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setCamera("left")
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setCamera("right")

# Setting node configs
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# Align depth map to the perspective of RGB camera, on which inference is done
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.setSubpixel(True)
stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("depth")
# Define a source - color camera
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setPreviewSize(NN_SIZE)
camRgb.setInterleaved(False)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
camRgb.setIspScale(1,3)
# camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
# camRgb.setPreviewKeepAspectRatio(False)

# Define the output streams
xoutRgb = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")

camRgb.video.link(xoutRgb.input)

yolo_spatial_det_nn = pipeline.createYoloSpatialDetectionNetwork()
# Define the model properties
yolo_spatial_det_nn.setConfidenceThreshold(0.5)
yolo_spatial_det_nn.setBlobPath(nnBlobPath)
yolo_spatial_det_nn.setNumClasses(9)  # Adjust based on your model
yolo_spatial_det_nn.setCoordinateSize(4)
yolo_spatial_det_nn.setAnchors([])  # Adjust based on your model
yolo_spatial_det_nn.setAnchorMasks({})  # Adjust based on your model
yolo_spatial_det_nn.setIouThreshold(0.5)
yolo_spatial_det_nn.setDepthLowerThreshold(100)
yolo_spatial_det_nn.setDepthUpperThreshold(5000)

xoutNN = pipeline.create(dai.node.XLinkOut)
xoutNN.setStreamName("detections")
yolo_spatial_det_nn.out.link(xoutNN.input)
camRgb.preview.link(yolo_spatial_det_nn.input)

passthroughOut = pipeline.create(dai.node.XLinkOut)
passthroughOut.setStreamName("pass")

stereo.depth.link(yolo_spatial_det_nn.inputDepth)
yolo_spatial_det_nn.passthroughDepth.link(xoutDepth.input)
yolo_spatial_det_nn.passthrough.link(passthroughOut.input)


# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    
    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    qPass = device.getOutputQueue(name="pass", maxSize=4, blocking=False)
    # qDet = device.getOutputQueue(name="nn")
    # startTime = time.monotonic()
    # counter = 0
    # fps = 0
    # color = (255, 255, 255)
    text = TextHelper(color=bbox_color, text_color=text_color)
    fps = FPSHandler()
    def get_center_point(bbox):
        xmin, ymin, xmax, ymax = bbox
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        return int(x_center), int(y_center)
    
    def draw_dynamic_line(frame, bbox, multiplier1, multiplier2):
        # Get frame dimensions
        height, width, _ = frame.shape

        # Calculate the center point of the bounding box
        xmin, ymin, xmax, ymax = bbox
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2

        # Define the starting point (mid bottom of the frame)
        start_point = (width // 2, height)

        # Calculate the direction vector
        direction_vector = np.array([x_center - start_point[0], y_center - start_point[1]])

        # Normalize the direction vector
        norm = np.linalg.norm(direction_vector)
        if norm != 0:
            direction_vector = direction_vector / norm

        # Calculate the dynamic line length
        line_length = multiplier1 * multiplier2

        # Calculate the new end point using the scaled direction vector
        end_point = (int(start_point[0] + direction_vector[0] * line_length),
                    int(start_point[1] + direction_vector[1] * line_length))

        # Draw the line on the frame
        cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        cv2.circle(frame, end_point,5, (255, 255, 255), 5, -1)
        return frame
 
    def draw_animation(frame, bbox,x_point, z_point, m_to_pix):

        x_meters = x_point
        z_meters = z_point
        # Get frame dimensions
        height, width, _ = frame.shape
        x_point = math.trunc(x_point*m_to_pix)
        z_point = math.trunc(z_point * m_to_pix)
        # Calculate the center point of the bounding box

        # Define the starting point (mid bottom of the frame)
        start_point = (width // 2, height)
        end_point = (start_point[0] + x_point, start_point[1] - z_point)

        print(start_point, end_point)
        # Draw the line on the frame
        cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        cv2.circle(frame, end_point,5, (255, 255, 255), 5, -1)
        text = f"X (meters): {x_meters}, Z(meters): {z_meters}"
        cv2.putText(frame, text, (end_point[0]+ 10, end_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3, cv2.LINE_AA)

        return frame   
    # Function to calculate radius
    def calculate_radius(speed):
        # Using log to calculate radius
        radius = constant / np.log(speed + 1)  # Add 1 to avoid log(0)
        # Clamping the radius to be within the min and max bounds
        return int(max(min(radius, max_radius), min_radius))
    
    # Function to draw a circular line
    def draw_circle(frame, radius):
        center = (frame.shape[1]//2 , frame.shape[0])
        # Drawing a circle in white color with 5 pixels thickness
        cv2.circle(frame, center, radius, (255, 255, 255), 5)


    
    def frameNorm(frame, NN_SIZE, bbox):
        # Check difference in aspect ratio and apply correction to BBs
        ar_diff = NN_SIZE[0] / NN_SIZE[0] - frame.shape[0] / frame.shape[1]
        sel = 0 if 0 < ar_diff else 1
        bbox[sel::2] *= 1-abs(ar_diff)
        bbox[sel::2] += abs(ar_diff)/2
        # Normalize bounding boxes
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(bbox, 0, 1) * normVals).astype(int)
    
    def displayFrame(name, frame, ani_frame):
        for detection in detections:

            
            bbox = frameNorm(frame, NN_SIZE, np.array([detection.xmin, detection.ymin, detection.xmax, detection.ymax]))
            text.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20))
            text.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40))
            text.putText(frame, f"Z: {int(detection.spatialCoordinates.z)/1000} m", (bbox[0] + 10, bbox[1] + 60))
            text.putText(frame, f"X: {int(detection.spatialCoordinates.x)/1000} m", (bbox[0] + 10, bbox[1] + 80))
            text.putText(frame, f"Y: {int(detection.spatialCoordinates.y)/1000} m", (bbox[0] + 10, bbox[1] + 100))

            # text.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (bbox[0] + 10, bbox[1] + 80))
            # text.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (bbox[0] + 10, bbox[1] + 100))

            text.rectangle(frame, bbox)
            # cv2.line(frame, (frame.shape[1]//2, frame.shape[0]), (get_center_point(bbox)),  (0, 255, 0), 2)
            # draw_dynamic_line(frame, bbox, int(detection.spatialCoordinates.z)/1000, METER_TO_PIXEL)
            draw_animation(frame, bbox, int(detection.spatialCoordinates.x)/1000,int(detection.spatialCoordinates.z)/1000, METER_TO_PIXEL)
            draw_animation(ani_frame, bbox, int(detection.spatialCoordinates.x)/1000,int(detection.spatialCoordinates.z)/1000, METER_TO_PIXEL)
            
            # cv2.line(frame, (frame.shape[1]//2, frame.shape[0]), (int((bbox[2]/bbox[0])//2), math.trunc((int(detection.spatialCoordinates.z)/1000)*METER_TO_PIXEL)),  (0, 255, 0), 2)
            # print((frame.shape[1]//2, frame.shape[0]), (int((bbox[2]/bbox[0])//2), math.trunc((int(detection.spatialCoordinates.z)/1000)*METER_TO_PIXEL)))
        # Show the frame
        cv2.imshow(name, frame)


    while True:
        inPreview = previewQueue.get()
        inDet = detectionNNQueue.get()
        depth = depthQueue.get()
        frame = inPreview.getCvFrame()
        depthFrame = depth.getFrame() # depthFrame values are in millimeters

        depth_downscaled = depthFrame[::4]
        if np.all(depth_downscaled == 0):
            min_depth = 0  # Set a default minimum depth value when all elements are zero
        else:
            min_depth = np.percentile(depth_downscaled[depth_downscaled != 0], 1)
        max_depth = np.percentile(depth_downscaled, 99)
        depthFrameColor = np.interp(depthFrame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
        if inDet is not None:
            detections = inDet.detections
            fps.next_iter()
        
        animmated_frame  = np.zeros(frame.shape, dtype=np.uint8)
        radius = calculate_radius(5)
        print("Radius", radius)
        draw_circle(animmated_frame, radius)
        text.putText(frame, "NN fps: {:.2f}".format(fps.fps()), (2, frame.shape[0] - 4))

        displayFrame("preview", frame, animmated_frame)

        cv2.imshow("Animated", animmated_frame)
        if cv2.waitKey(1) == ord('q'):
            break
