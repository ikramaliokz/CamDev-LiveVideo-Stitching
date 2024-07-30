NN_SIZE = (300, 300)
METER_TO_PIXEL = 50  # value for one meter length in pixels
min_radius = 50
max_radius = 800
constant = 800

nnBlobPath = "models/mobilenet-ssd_openvino_2021.2_6shave.blob"
text_color = (0, 0, 255)
bbox_color = (0, 255, 0)


# MobilenetSSD label texts
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

