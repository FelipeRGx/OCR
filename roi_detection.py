import cv2
import numpy as np
from scipy import stats as st
from dataclasses import dataclass
import checkbox_detector.detection_main as detectors
import sys
sys.path.append('checkbox_detector/')
import matplotlib.pyplot as plt

colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (128, 0, 128),
]


@dataclass
class RegionOfInterest(object):
    '''
    Class to represent a region of interest
    
    Attributes
    ----------
    x : int
        x coordinate of the top left corner of the region
        
    y : int
        y coordinate of the top left corner of the region
    
    w : int
        width of the region
    
    h : int
        height of the region
    
    id : int
        id of the region
    
    Methods
    -------
    intersects(other)
        Returns True if the region intersects with the other region
    '''

    x: int = -1
    y: int = -1
    w: int = -1
    h: int = -1
    id: int = -1

    def __hash__(self):
        return hash((self.x, self.y, self.w, self.h))

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.__dict__ == other.__dict__

    def intersects(self, other):
        x_overlap = max(
            0, min(self.x + self.w, other.x + other.w) - max(self.x, other.x)
        )
        y_overlap = max(
            0, min(self.y + self.h, other.y + other.h) - max(self.y, other.y)
        )
        return x_overlap * y_overlap > 0


def get_document_sections(img, conf=0.4 ,show=False):
    image = img.copy()
    if len(image.shape) < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # get the document sections
    detections = detectors.section_detector.detect(image, iou=0.1, conf=conf)
    detections = sort_by_center(detections[0].detach().numpy())

    # draw the bounding box
    for i, detection in enumerate(detections):
        x0, y0, x1, y1, conf, cls = detection

        # draw the bounding box
        cv2.rectangle(
            image, (int(x0), int(y0)), (int(x1), int(y1)), colors[int(cls)], 2 # type: ignore
        )
        # draw the label
        cv2.putText(
            image,
            str(i) + " " + detectors.section_detector.names[int(cls)] + " " + str(conf), 
            (int(x0), int(y0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            colors[int(cls)],
            1,
            cv2.LINE_AA,
        )

    # show the output image
    if show:
        plt.imshow(image)
        plt.show()

    return sort_by_center(detections)

def get_checkboxes(img, show=False):
    image = img.copy()
    if len(image.shape) < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # get the document sections
    detections = detectors.checkbox_detector_2.detect(image, conf=.01)
    detections = sort_by_center(detections[0].detach().numpy())

    # draw the bounding box
    for i, detection in enumerate(detections):
        x0, y0, x1, y1, conf, cls = detection
        if(conf > 0.1):
            # draw the bounding box
            cv2.rectangle(
                image, (int(x0), int(y0)), (int(x1), int(y1)), colors[int(cls)], 1
            )
            # draw the label
            cv2.putText(
                image,
                str(i) , # type: ignore
                (int(x0), int(y0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                colors[int(cls)],
                1,
                cv2.LINE_AA,
            )

    # show the output image
    if show:
        plt.imshow(image)
        plt.show()

    return detections

def sort_by_center(objects):
    # Calculate centroids for each object
    centroids = []
    for obj in objects:
        x0, y0, x1, y1 = obj[:4]
        centroid_x = (x0 + x1) / 2
        centroid_y = (y0 + y1) / 2
        centroids.append((centroid_x, centroid_y))

    # Sort the objects based on the centroids and their appearance from left to right and top to bottom
    sorted_objects = [
        obj
        for _, obj in sorted(zip(centroids, objects), key=lambda x: (x[0][1], x[0][0]))
    ]

    return sorted_objects
