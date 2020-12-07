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
import time

MODEL_PATH = './model_weights/saved_model'
FACE_DB = './face_db/face_db.npy'
NAME_DB = './face_db/name_db.npy'


class InsightFace:

    def __init__(self):
        self.retinaface = RetinaFace(score_thres=0.8, iou_thres=0.45)
        self.model_path = MODEL_PATH
        self.predictor = self.get_insightface().signatures['a_signature']
        # TODO: use grid search to optimize this parameter
        self.dis_thres = 1.0

    def get_insightface(self):
        predictor = tf.saved_model.load(self.model_path)
        return predictor

    def get_face_db(self):
        face_db = np.load(FACE_DB)
        name_db = np.load(NAME_DB)

        return [face_db, name_db]

    def recognize(self, img, detect_vis=False):

        face_db, name_db = self.get_face_db()

        original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_copy = original_img.copy()  # RBG format

        # start1 = time.time()
        boxes, total_landmarks = self.retinaface.predict(original_img, visualize=detect_vis)
        # end1 = time.time()
        # print('retina face spends {}'.format(end1 - start1))

        # get reference face points
        reference_points = get_reference_facial_points(default_square=True)

        if len(boxes) > 0:

            face_ids = []
            distance_list = []

            for i, box in enumerate(boxes):
                landmarks = total_landmarks[i]  # (num_bbox, 10[x1, x2, x3, x4, x5, y1, y2, y3, y4, y5])
                landmarks = np.array([[landmarks[j], landmarks[j + 5]] for j in range(5)])
                rotated_img = warp_and_crop_face(img_copy, landmarks, reference_points, crop_size=(112, 112))
                # cv2.imshow('rotated', rotated_img)
                # cv2.waitKey(0)
                # cv2.imwrite('img/shu_rot.jpg', cv2.cvtColor(rotated_img, cv2.COLOR_RGB2BGR))

                # preprocess feed image
                rotated_img = preprocess(rotated_img)
                # start2 = time.time()
                results = self.predictor(inputs=rotated_img, drop=tf.constant(1.0))['outputs']
                # end2 = time.time()
                # print('insight face spends {}'.format(end2 - start2))
                distances = get_distance(results, face_db)
                similar_id, min_dis = get_similar_face(distances, dis_thres=self.dis_thres)

                face_id = name_db[similar_id]
                distance_list.append(min_dis)

                if not face_id:
                    face_id = 'stranger'
                face_ids.append(face_id)

            return [face_ids, boxes, total_landmarks, distance_list]
        else:
            # print('Do not detect out any face in picture!')
            return None


if __name__ == '__main__':
    img_path = 'img/shu.jpg'
    original_img = cv2.imread(img_path)
    insight_face = InsightFace()
    for i in range(10):
        face_name, _, _, _ = insight_face.recognize(original_img)
        print(face_name)
