# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:recognize.py
# software: PyCharm

from retinaface.predict import RetinaFace
import cv2
from alignment.align_trans import warp_and_crop_face
from alignment.align_trans import get_reference_facial_points
import numpy as np
import tensorflow as tf
from insightface.util.utils import preprocess, get_distance, get_similar_face
import json

MODEL_PATH = './model_weights/saved_model'
FACE_DB = './face_db/face_db.npy'
NAME_DB = './face_db/name_db.npy'


class InsightFace:

    def __init__(self):
        self.retinaface = RetinaFace(score_thres=0.5, iou_thres=0.45)
        self.model_path = MODEL_PATH
        self.predictor = self.get_insightface().signatures['a_signature']
        # TODO: use grid search to optimize this parameter
        self.dis_thres = 0.9

    def get_insightface(self):
        predictor = tf.saved_model.load(self.model_path)
        return predictor

    def get_face_db(self):
        face_db = np.load(FACE_DB)
        name_db = np.load(NAME_DB)

        return [face_db, name_db]

    def recognize(self, file_path, detect_vis=False, recog_vis=True):

        face_db, name_db = self.get_face_db()

        original_img = cv2.imread(file_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        img_copy = original_img.copy()  # RBG format

        boxes, total_landmarks = self.retinaface.predict(original_img, visualize=detect_vis)

        # get reference face points
        reference_points = get_reference_facial_points(default_square=True)

        if len(boxes) > 0:

            face_ids = []

            for i, box in enumerate(boxes):
                landmarks = total_landmarks[i]  # (num_bbox, 10[x1, x2, x3, x4, x5, y1, y2, y3, y4, y5])
                landmarks = np.array([[landmarks[j], landmarks[j + 5]] for j in range(5)])
                rotated_img = warp_and_crop_face(img_copy, landmarks, reference_points, crop_size=(112, 112))
                # cv2.imshow('rotated', rotated_img)
                # cv2.waitKey(0)
                # cv2.imwrite('img/techi_rot.jpg', rotated_img)

                # preprocess feed image
                rotated_img = preprocess(rotated_img)
                results = self.predictor(inputs=rotated_img, drop=tf.constant(1.0))['outputs']
                distances = get_distance(results, face_db)
                similar_id = get_similar_face(distances, dis_thres=self.dis_thres)
                face_id = name_db[similar_id]
                face_ids.append(face_id)

            return face_ids
        else:
            print('Do not detect out any face in picture!')
            return None


if __name__ == '__main__':
    img_path = 'img/woman.jpg'
    insight_face = InsightFace()
    face_name = insight_face.recognize(img_path)
    print(face_name)
