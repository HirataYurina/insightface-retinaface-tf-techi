# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:test_saved_model.py
# software: PyCharm

import tensorflow as tf
import cv2
from insightface.util.utils import preprocess
import numpy as np

# ###################################
# test the saved_model and is right
# ###################################
predictor = tf.saved_model.load('./model_weights/saved_model').signatures['a_signature']

face1 = cv2.imread('img/woman.jpg')
face2 = cv2.imread('face_db/face_database/woman_res.jpg')
face1 = cv2.cvtColor(face1, cv2.COLOR_BGR2RGB)
face2 = cv2.cvtColor(face2, cv2.COLOR_BGR2RGB)

face1 = preprocess(face1)
face2 = preprocess(face2)

face1_embedding = predictor(inputs=face1,
                            drop=tf.constant(1.0))['outputs']
face2_embedding = predictor(inputs=face2,
                            drop=tf.constant(1.0))['outputs']

face1_embedding = face1_embedding / np.linalg.norm(face1_embedding)
face2_embedding = face2_embedding / np.linalg.norm(face2_embedding)

dis = np.sum(np.square(face1_embedding - face2_embedding))

print(dis)
