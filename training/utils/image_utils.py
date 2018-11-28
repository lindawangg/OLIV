from object_detection.utils.visualization_utils import draw_bounding_box_on_image_array
import cv2
import csv
from collections import defaultdict

class ImageContainer(object):
    '''
        Generic class for a single image.
    '''
    def __init__(self, file_name):
        self.file_name = file_name
        self.image = None
        self.width = None
        self.height = None
        self.detected_objects = []
        self.labelled_objects = []
        self.ious = []
        self.precision = []
        self.read_image()

    def read_image(self):
        if not self.image:
            try:
                self.image = cv2.imread(self.file_name)
                self.height, self.width, _ = self.image.shape
            except Exception as e:
                print e

    def __str__(self):
        return self.file_name

    def save_image(self):
        cv2.imwrite(self.file_name, self.image)

    def draw_boxes(self, thickness=3):
        if not self.detect_objects:
            for box in self.labelled_objects:
                box.draw_bounding_box(self.image)
            for box in self.detected_objects:
                box.draw_bounding_box(self.image, det_object=True)

class ImageObject(object):
    def __init(self, width, height, obj_type, xmin, ymin, xmax, ymax):
        self.obj_type = obj_type
        self.height = int(height)
        self.width = int(width)
        self.xmin = int(xmin)
        self.xmax = int(xmax)
        self.ymin = int(ymin)
        self.ymax = int(ymax)

        self.verify_coords()

    def __eq__(self, obj):
        return self.obj_type == obj.obj_type and self.xmin == obj.xmin \
            and self.xmax == obj.xmax and self.ymin == obj.ymin \
            and self.ymax == obj.ymax

    def __cmp__(self, obj):
        if self.obj_type != obj.obj_type:
            return float('inf')

    def verify_coords(self):
        if self.xmax < self.xmin:
            raise ValueError('xmax for a bounding box must be greater than xmin')

        if self.ymax < self.ymin:
            raise ValueError('ymax for a bounding box must be greater than ymin')

        if self.height and self.ymax > self.height:
            raise ValueError('ymax for bounding box has to be less than height of image')

        if self.width and self.xmax > self.width:
            raise ValueError('xmax for bounding box has to be less than height of image')

    def __str__(self):
        return '%s::%s,%s,%s,%s' % (self.obj_type, self.xmin, self.xmax, self.ymin, self.ymax)

    def draw_bounding_box(self, img, det_object=False):
        if det_object:
            color = 'green'
        else:
            color = 'white'
        draw_bounding_box_on_image_array(img, self.ymin, self.xmin, self.ymax, self.xmax, color=color, thickness=6, display_str_list=(self.obj_type, ), use_normalized_coordinates=False)

def build_labelled_csv_dictionary(csv_file_name):
    res = defaultdict(list)

    with open(csv_file_name) as csv_file:
        csvreader = csv.reader(csv_file)
        for row in csvreader:
            if 'filename' == row[0]:
                continue
            res[row[0]].append(ImageObject(*row[1:]))
    return res
