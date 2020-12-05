# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:predict.py
# software: PyCharm

from retinaface.net.face import retinaface_net
import tensorflow.keras as keras
import numpy as np
from retinaface.util.utils import generate_anchors, delta2box, delta2landmarks, nms


class RitinaFace:

    def __init__(self, score_thres=0.4):
        self.score_thres = score_thres
        self.generate()

    def generate(self):
        # generate the model when initiate RetinaFace
        inputs = keras.Input(shape=(None, None, 3))
        self.retinaface = retinaface_net(inputs=inputs,
                                         training=False)
        self.retinaface.load_weights('./model_weights/retinaface_resnet50.h5', by_name=True)

    def predict(self, image):

        height = image.shape[0]
        width = image.shape[1]

        img_copy = image.copy()

        image = keras.applications.imagenet_utils.preprocess_input(image)
        image = np.expand_dims(image, axis=0)

        raw_pred = self.retinaface(image)

        # decode raw prediction
        bbox, classification, landmarks = raw_pred
        anchors = generate_anchors(height, width)
        bbox = delta2box(bbox, anchors)  # corner points
        landmarks = delta2landmarks(landmarks, anchors)

        # normalize coordinate to original image coordinate
        scale_bbox = np.tile([width, height], reps=2)
        scale_landmarks = np.repeat([width, height], axis=0, repeats=[5, 5])

        bbox = bbox * scale_bbox
        landmarks = landmarks * scale_landmarks

        # start NMS
        results = nms()

