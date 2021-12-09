import os
import logging
import numpy as np
import insightface
import cv2
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image


class Detector:
    classes = {
        "male": [1],
        "female": [0],
    }
    # rpn_threshold = 0.5

    def __init__(self):
        self.logger = logging.getLogger("face_detector")
        handler = logging.NullHandler()
        self.logger.addHandler(handler)

        self.app = FaceAnalysis()
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        self.logger.info("Face detector initialized")

        self.i = 0


    def run_inference_for_single_image(self, img):

        faces = self.app.get(img)
        num_detections = len(faces)
        
        # if num_detections > 0:
        #     rimg = self.app.draw_on(img, faces)
            # print('Detected faces. Writing out.')
            # plt.imshow(rimg)
            # plt.show()
            # cv2.imwrite("~/ZQ/dds/backend/temp_f/" + str(self.i) + ".jpg", rimg)
            # self.i += 1
        

        output_dict = {}
        detection_boxes = []
        detection_scores = []
        detection_classes = [] # gender
        detection_ages = []

        for face in faces:
            detection_boxes.append(face['bbox'])
            detection_scores.append(face['det_score'])
            detection_classes.append(face['gender']) 
            detection_ages.append(face['age'])
            
        
        output_dict['num_detections'] = int(np.array(num_detections))
        output_dict['detection_classes'] = np.array(detection_classes).astype(np.uint8)
        output_dict['detection_boxes'] = np.array(detection_boxes)
        output_dict['detection_scores'] = np.array(detection_scores)
        output_dict['detection_ages'] = np.array(detection_ages)

        return output_dict



    def infer(self, image_np):
        image_crops = image_np

        # this output_dict contains both final layer results and RPN results
        output_dict = self.run_inference_for_single_image(image_crops)

        # The results array will have (class, (xmin, xmax, ymin, ymax)) tuples
        results = []
        for i in range(len(output_dict['detection_boxes'])):
            object_class = output_dict['detection_classes'][i]
            relevant_class = False
            for k in Detector.classes.keys():
                if object_class in Detector.classes[k]:
                    object_class = k
                    relevant_class = True
                    break
            if not relevant_class:
                continue

            xmin, ymin, xmax, ymax = output_dict['detection_boxes'][i]
            # ymin, xmin, ymax, xmax = output_dict['detection_boxes'][i]

            confidence = output_dict['detection_scores'][i]
            box_tuple = (xmin, ymin, xmax - xmin, ymax - ymin)
            results.append((object_class, confidence, box_tuple))

        return results, [] # empty rpn regions
