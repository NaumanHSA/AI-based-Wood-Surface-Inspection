import os

########################################################### Genreal Parameters #####################################################

# SOURCE is the input path to process. It can be path to an image, a video or a directory containing images. If it's value is set
# to 0, then it will operate on webcam.
SOURCE = "test" # process images
# SOURCE = 0    # webcam

# the results path is where all the output files are saved
RESULTS_PATH = "runs/Experiment" # save results in this path

# path to the trained weights file (.pt)
PRETRAINED_WEIGHT_PATH = "weights/wood_surface_inspection.pt"

# whether to preview results during inference or not?
VISUALIZE = False

# whether to save annotations (txt files) and images?
SAVE_ANNOTATIONS = True

# whether to save the visualizations shown in preview?
SAVE_VISUALIZATIONS = True

# # step size (number of frames) after which a record be added to the DB to ensure no overlap
# FRAMES_STEP = 10


####################################################### YOLOv5 Model Parameters ####################################################

# input size of the image (the size on which the data was trained)
IMAGE_SIZE = [608]  # inference size (pixels)
IMAGE_SIZE *= 2 if len(IMAGE_SIZE) == 1 else 1

# confidence threshold below which the objects will get discarded. (every detection has confidence score between 0 and 1)
CONFIDENCE_THRESHOLD = 0.5  # confidence threshold

# IoU (intersection over union area) threshold below which the objects will get discarded. 
# (every detection has iou score between 0 and 1)
IOU_THRESHOLD = 0.4  # NMS IOU threshold

BBOX_LINE_THICKNESS = 2  # bounding box thickness (pixels)


####################################################### Measurment Parameters ####################################################

FOA_HORIZONTAL = 91.44 # horizontal field of view (horizontal area covered by the camera view in centimeters)
FOA_VERTICAL = 60.96  # vertical field of view (vertical covered by the camera view in centimeters)


####################################################### Image Cropping Parameters ###################################################

LEFT_CROP_PERCENTAGE = 25     # crop from the left (value in percentage with respect to the image width)
RIGHT_CROP_PERCENTAGE = 75    # crop from the right (value in percentage with respect to the image width)
TOP_CROP_PERCENTAGE = 25      # crop from the top (value in percentage with respect to the image height)
BOTTOM_CROP_PERCENTAGE = 75   # crop from the bottom (value in percentage with respect to the image height)

assert 0 <= LEFT_CROP_PERCENTAGE <= 100, "LEFT_CROP_PERCENTAGE always be between 0 and 100"
assert 0 <= RIGHT_CROP_PERCENTAGE <= 100, "RIGHT_CROP_PERCENTAGE always be between 0 and 100"
assert 0 <= TOP_CROP_PERCENTAGE <= 100, "TOP_CROP_PERCENTAGE always be between 0 and 100"
assert 0 <= BOTTOM_CROP_PERCENTAGE <= 100, "BOTTOM_CROP_PERCENTAGE always be between 0 and 100"
assert LEFT_CROP_PERCENTAGE < RIGHT_CROP_PERCENTAGE, "LEFT_CROP_PERCENTAGE always be smaller then RIGHT_CROP_PERCENTAGE"
assert TOP_CROP_PERCENTAGE < BOTTOM_CROP_PERCENTAGE, "TOP_CROP_PERCENTAGE always be smaller then BOTTOM_CROP_PERCENTAGE"
####################################################################################################################################


####################################################### Database Parameters ###################################################

SERVER_IP = "NOMI-PC\SQLEXPRESS"   # Server IP or name of the server
DATABASE = "WOOD_INSPECTION"    # Database name where the table exists
USER = ""   # Username
PASSWORD = ""   # Passwords

TABLE_NAME = "Detections"   # Table name where to store data

# insertion query
INSERT_QUERY = f"""
    INSERT INTO {TABLE_NAME} 
    (
        ImageName, 
        ImageWidth, 
        ImageHeight, 
        DefectType, 
        DefectWidth, 
        DefectHeight, 
        ConfidenceScore, 
        NormBoxX, 
        NormBoxY,
        NormBoxWidth, 
        NormBoxHieght
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""
####################################################################################################################################


