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
import cv2


class RetinaFace:

    def __init__(self, score_thres=0.5, iou_thres=0.45):
        self.score_thres = score_thres
        self.iou_thres = iou_thres
        self.generate()

    def generate(self):
        # generate the model when initiate RetinaFace
        inputs = keras.Input(shape=(None, None, 3))
        self.retinaface = retinaface_net(inputs=inputs,
                                         training=False)
        self.retinaface.load_weights('../retinaface/model_weights/retinaface_resnet50.h5', by_name=True)

    def predict(self, image, visualize=True):

        height = image.shape[0]
        width = image.shape[1]

        img_copy = image.copy()
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)

        image = keras.applications.imagenet_utils.preprocess_input(image)
        image = np.expand_dims(image, axis=0)

        raw_pred = self.retinaface(image)  # a eager tensor

        # #########################################################################
        # decode raw prediction
        bbox, classification, landmarks = raw_pred

        bbox = np.squeeze(bbox)
        classification = np.squeeze(classification)
        landmarks = np.squeeze(landmarks)

        anchors = generate_anchors(height, width)
        bbox = delta2box(bbox, anchors)  # corner points
        landmarks = delta2landmarks(landmarks, anchors)

        # normalize coordinate to original image coordinate
        scale_bbox = np.tile([width, height], reps=2)
        scale_landmarks = np.repeat([width, height], axis=0, repeats=[5, 5])

        bbox = bbox * scale_bbox
        landmarks = landmarks * scale_landmarks  # (num_bbox, 10[x1, x2, x3, x4, x5, y1, y2, y3, y4, y5])

        # start NMS
        classification = classification[..., 1]  # (neg, pos) so pick index=1
        scores, regression, points = nms(classification, bbox, landmarks,
                                         score_thres=self.score_thres, iou_thres=self.iou_thres)

        # draw faces in original image
        if len(scores) > 0 and visualize:
            for i, score in enumerate(scores):
                box = regression[i]
                point = points[i]
                # draw box
                cv2.rectangle(img_copy,
                              (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])),
                              color=(255, 0, 0),
                              thickness=1)

                # write score
                text_point = (int(box[0]), int(box[1]))
                cv2.putText(img_copy, '{:.3f}'.format(score),
                            org=text_point,
                            fontFace=cv2.FONT_HERSHEY_DUPLEX,
                            fontScale=0.5,
                            color=(255, 255, 255))

                # draw landmarks
                for j in range(5):
                    p = (int(point[j]), int(point[j + 5]))
                    cv2.circle(img_copy,
                               center=p,
                               radius=2,
                               color=(0, 255, 255),
                               thickness=-1)
            cv2.imwrite('img/techi_res.jpg', img_copy)
            cv2.imshow('result', img_copy)
            cv2.waitKey(0)

        return regression, points


if __name__ == '__main__':

    # #################################################
    # This is a example to predict your own image
    techi = cv2.imread('./img/techi.jpg')

    # cv2.imread is BGR model, change it to RGB
    techi = cv2.cvtColor(techi, cv2.COLOR_BGR2RGB)

    # get a predictor
    predictor = RetinaFace()
    # start predicting
    predictor.predict(techi)
    # #################################################
