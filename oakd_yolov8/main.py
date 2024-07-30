import numpy as np
import cv2
import depthai as dai
from pipeline_setup import setup_pipeline
from utils import TextHelper, FPSHandler, displayFrame
from config import bbox_color, text_color, NN_SIZE, nnBlobPath


def process_frames(device, radius1, radius2, radius3):
    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    qPass = device.getOutputQueue(name="pass", maxSize=4, blocking=False)
    text = TextHelper(color=bbox_color, text_color=text_color)
    fps = FPSHandler()
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
        # radius = calculate_radius(5)

        # draw_circle(animmated_frame, radius1, radius2, radius3)
        text.putText(frame, "NN fps: {:.2f}".format(fps.fps()), (2, frame.shape[0] - 4))

        displayFrame("preview", detections, frame, animmated_frame, text, radius1, radius2, radius3 )

        cv2.imshow("Animated", animmated_frame)
        if cv2.waitKey(1) == ord('q'):
            break

def main():

    pipeline = setup_pipeline(NN_SIZE, nnBlobPath)
    radius1 = 50
    radius2 = 100
    radius3 = 150
    with dai.Device(pipeline) as device:

        process_frames(device, radius1, radius2, radius3)

if __name__ == '__main__':
    main()
