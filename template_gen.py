import preprocessing
import roi_detection
import cv2
import os
from text_detection import CraftTextDetector, GoogleCloudVision
import numpy as np
import easyocr

class TemplateUtil(object):
    """docstring for TemplateUtil."""
    def __init__(self, processor = None, model = None):
        super(TemplateUtil, self).__init__()
        self.text_detector = CraftTextDetector(None)
        self.text_recognition = GoogleCloudVision()
        
        self.regions_of_interest = {}
        self.recognizer = easyocr.Reader(['es'], gpu=False)
        if processor is not None and model is not None:
            self.processor = processor
            self.model = model

    def check_if_exist(self, path):
        return NotImplemented

    def preprocess_template_page(self, img):
        blur = preprocessing.gaussian_blur(img)
        th = preprocessing.adaptive_threshold(blur)
        result = preprocessing.open_operation(th)
        mask_applied = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) - result
        output = preprocessing.brute_noise_supression(mask_applied)
        output = preprocessing.crop_active_area(output)
        # output = preprocessing.pad_img(output)
        return output

    def get_data_fields_names(self, path):
        return None

    def get_regions_of_interest(self, img, show=False):
        roi_detection.get_regions_of_interest(img)
        self.regions_of_interest = roi_detection.regions_of_interest_detected
    
    def get_template_text(self, img):
        self.get_regions_of_interest(img.copy())
        img_without_checkbox = self.create_image_without_rois(img.copy(), self.regions_of_interest["checkboxes"])
        text = self.text_detector.detect(img_without_checkbox)
        for i, box in enumerate(text['boxes']):
            box = np.array(box).astype(np.int32).reshape((-1))
            box = box.reshape(-1, 2)
            crop_img = img[box[0][1]:box[2][1], box[0][0]:box[2][0]]

            texto = self.img2text_gcp(crop_img)
        return text
    
    def img2Text(self, img):
        # path to model
        pixel_values = self.processor(images=img, return_tensors="pt").pixel_values

        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print("Texto de TrOCR: ", generated_text)
        generated_text = self.recognizer.readtext(img, detail=0)
        print(generated_text)
        return generated_text
    
    def img2text_gcp(self, img):
        response = self.text_recognition.recognise_text(img)
        # self.text_recognition.print_text(response)
        print(self.text_recognition.get_text(response))
        return response
    
    def create_image_without_rois(self, img, rois):
        for roi in rois:
            img[roi.y-2:roi.y+roi.h+2, roi.x-2:roi.x+roi.w+2] = 255
        return img

    def create_template(self, path):
        pages = preprocessing.get_imgs_from_path(path, False)

        if not os.path.exists("templates/"):
            os.makedirs("templates/")

        list_id = os.listdir("templates")

        if not os.path.exists("templates/"):
            os.makedirs("templates/")

        if len(list_id) == 0:
            # create first template
            max_id = 0
            os.makedirs("templates/1")
        else:
            # create next template
            max_id = list_id[-1]
            os.makedirs("templates/" + str(int(max_id) + 1))

        for i, page in enumerate(pages):
            page = self.preprocess_template_page(page)
            cv2.imwrite(
                "templates/" + str(int(max_id) + 1) + "/" + str(i) + ".png", page
            )
            # detection.get_regions_of_interest_from_template(page, True)

