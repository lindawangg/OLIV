import os
import sys

CWD_PATH = os.getcwd()

# Path to tensorflow model
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
LABEL_MAP_NAME = 'mscoco_label_map.pbtxt'
PATH_TO_CKPT = os.path.join(CWD_PATH, 'models', MODEL_NAME, 'frozen_inference_graph.pb')

# Path to labels
PATH_TO_LABELS = os.path.join(CWD_PATH, 'models', MODEL_NAME, LABEL_MAP_NAME)

DETECT_THRESHOLD = 0.4

NUM_CLASSES = 90

ALLOWED_CLASSES = ['bottle','knife','spoon','fork','cup','bowl','dog', 'kettle', 'cell phone', 'laptop', 'keyboard']
REFERENCE_OBJECTS = ['laptop','keyboard']

