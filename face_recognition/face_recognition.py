import os
import onnxruntime as ort
import random
import numpy as np
import cv2
import time
import torch
import torchvision
from face_recognition.iresnet import iresnet100
from face_recognition.utils import Utils
import math

class FaceRecognition:
    def __init__(self,
                 detection_weights = './weights/detection_face.onnx',
                 conf_thres=0.7,
                 iou_thres=0.5,
                 classes_txt='./face_recognition/classes.txt',
                 face_feat_extraction_weights='./weights/face_feat_extraction.onnx',
                 database_tensor='./face_database'):
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device()=='GPU' else ['CPUExecutionProvider']
        self.detection_weights=detection_weights
        self.conf_thres=conf_thres
        self.iou_thres=iou_thres
        self.names=classes_txt
        self.recog_session=None
        self.detect_session=None
        self.face_feat_extraction_weights=face_feat_extraction_weights
        self.database_tensor=database_tensor
        self.detect_session=ort.InferenceSession(self.detection_weights, providers=self.providers)
        self.recog_session=ort.InferenceSession(self.face_feat_extraction_weights, providers=self.providers)

    def draw_bbox(self, img, box, text, similarity):
        similarity = str(round(similarity, 3))
        color = [255, 255, 0]
        score = box[-1]
        box = box[:-1]
        start_point = (box[:2])
        end_point = (box[2:])

        cv2.rectangle(img, box[:2], box[2:], color, 2)
        cv2.putText(img, text+'  '+similarity, (box[0], box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [255, 255, 0], thickness=2)
        return img

    def draw_detect_result(self, img_or, detected, names=None):
        img = img_or.copy()
        if detected is None:
            return img
        for i, (x0, y0, x1, y1, score) in enumerate(detected):
            if names is not None and names != False:
                # print(names)
                if i >= len(names):
                    continue
                img = self.draw_bbox(img, [x0, y0, x1, y1, score], names[i][0], names[i][1][0][0])
            else: 
                img = self.draw_bbox(img, [x0, y0, x1, y1, score], 'face')
        return img
    

    def detect(self, img_or):
        image, ratio, dwdh = Utils.letterbox(im=img_or, auto=False)
        image = image.transpose((2,0,1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
        im = image.astype(np.float32)
        im /= 255

        session = self.detect_session
        outname = [i.name for i in session.get_outputs()]
        inname = [i.name for i in session.get_inputs()]
        inp = {inname[0]: im}
        inp = np.array(inp[inname[0]], dtype=np.float32)

        if self.providers == ['CUDAExecutionProvider', 'CPUExecutionProvider']:
            X_ortvalue = ort.OrtValue.ortvalue_from_numpy(inp, 'cuda', 0)
            Y_ortvalue = ort.OrtValue.ortvalue_from_shape_and_type([1,25200,16], np.float32, 'cuda', 0)  # Change the shape to the actual shape of the output being bound
            io_binding = session.io_binding()
            io_binding.bind_input(name='input', device_type=X_ortvalue.device_name(), device_id=0, element_type=np.float32, shape=X_ortvalue.shape(), buffer_ptr=X_ortvalue.data_ptr())
            io_binding.bind_output(name='output', device_type=Y_ortvalue.device_name(), device_id=0, element_type=np.float32, shape=Y_ortvalue.shape(), buffer_ptr=Y_ortvalue.data_ptr())
            session.run_with_iobinding(io_binding)
            outputs = io_binding.get_outputs()[0]
            
            outputs = outputs.numpy()
        else:
            X_ortvalue = ort.OrtValue.ortvalue_from_numpy(inp, 'cpu')
            Y_ortvalue = ort.OrtValue.ortvalue_from_shape_and_type([1, 25200, 16], np.float32, 'cpu')
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            input_data = {input_name: X_ortvalue}
            outputs = session.run([output_name], input_data)[0]

        output = torch.from_numpy(outputs)

        detected = Utils.non_max_suppression_face(output, self.conf_thres, self.iou_thres) # Single face detection

        detected = detected[0]    
        if detected.nelement() == 0:
            return None
        else:
            bboxes = []
            for i, (x0, y0, x1, y1, score) in enumerate(detected[:,0:5]):
                box = np.array([x0, y0, x1, y1])
                box -= np.array(dwdh*2)
                box /= ratio
                box = box.round().astype(np.int32).tolist()
                score = round(float(score), 3)
                color = [255, 255, 0]
                box.append(score)
                bboxes.append(box)
            return bboxes
    
    def get_face_features(self, img, bboxes, get_cropped=False):
        """
        This method take original image and bounding boxes as inputs.
        Returning the features extracted from those bounding boxes
        [[feature1], 
        [feature2],
        [feature3]]
        """
        features = []

        cropped_boxes, imgs = Utils.crop_image_feat_extraction(img, bboxes, True)

        if cropped_boxes is None:
            return None

        for cropped_box in cropped_boxes:
            session = self.recog_session
            outname = [i.name for i in session.get_outputs()]
            inname = [i.name for i in session.get_inputs()]

            inp = {inname[0]:np.array(cropped_box)}

            outputs = session.run(outname, inp)[0]

            feature = torch.from_numpy(outputs)
            features.append(feature)
        if get_cropped == True:
            return features, imgs
        return features

    def recognize(self, img, bboxes):
        """
        This method take original image and bounding boxes as inputs. Returning 
        list of name and final_cosine:
        [[name1, final_cosine1], 
        [name2, final_cosine2],
        [name3, final_cosine3]]
        """
        features = self.get_face_features(img, bboxes)
        if features is None:
            return None
        results = []
        ts = os.listdir(f'{self.database_tensor}')
        # lc = 0
        lcc = 0
        w = []
        for feat in features:
            for i in range(len(ts)):
                tmp = np.load(f'{self.database_tensor}/{ts[i]}', allow_pickle=True)
                w.append(tmp)
                similarity = Utils.cosine_similarity(feat, tmp)
                if similarity >= Utils.cosine_similarity(feat, w[lcc]):
                    lcc = i
                    final_cosine = similarity

            # print(type(final_cosine))
            if final_cosine < 0.3:
                name = "Unknown"
            else:
                name = ts[lcc].replace(".npy", "")
            feat_name = [name, final_cosine]
            results.append(feat_name)
        return results
        
    def add_face(self, img, bboxes, name):

        features, cropped_boxes = self.get_face_features(img, bboxes, True)
        np.save(f'{self.database_tensor}/{name}.npy', features[0])
        return True, cropped_boxes