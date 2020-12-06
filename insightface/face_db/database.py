# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:database.py
# software: PyCharm

import os
import cv2
from insightface.util.utils import preprocess
import tensorflow as tf
import json
import numpy as np


DB_PATH = './face_database'
EMB_FILE = './face_db.npy'
NAME_FILE = './name_db.npy'


class FaceDataBase:

    def __init__(self):
        self.insightface = tf.saved_model.load('../model_weights/saved_model').signatures['a_signature']

    def add_face(self):
        # TODO: write this function
        pass

    def initialize_db(self):
        img_list = os.listdir(DB_PATH)

        name_list = []
        embedding_list = []

        for img in img_list:
            if img.endswith(('jpg', 'jpeg', 'png')):
                face_name = img.split('.')[0]

                # read img and embedding it
                image = cv2.imread(os.path.join(DB_PATH, img))

                if image.shape[0] != 112 or image.shape[1] != 112:
                    assert ValueError, "image shape is not (112, 112)"

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = preprocess(image)
                embedding = self.insightface(inputs=image, drop=tf.constant(1.0))['outputs']
                embedding = embedding.numpy()
                embedding_list.append(embedding)
                name_list.append(face_name)

        names = np.array(name_list)
        embeddings = np.concatenate(embedding_list, axis=0)

        # save in npy file
        np.save(EMB_FILE, embeddings)
        np.save(NAME_FILE, names)


if __name__ == '__main__':

    db = FaceDataBase()
    db.initialize_db()

    names = np.load(NAME_FILE)
    print(names.shape)
    embs = np.load(EMB_FILE)
    print(embs.shape)
