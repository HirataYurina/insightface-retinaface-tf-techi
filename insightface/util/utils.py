# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:utils.py
# software: PyCharm

import tensorflow as tf
import numpy as np


def preprocess(img):
    """ The feed in saved_model must be float tensor

    Args:
        img: array (112, 112, 3)

    Returns:
        a float tensor (1, 112, 112, 3)

    """

    img = (img - 127.5) / 127.5
    img = tf.constant(img, dtype=tf.float32)
    img = tf.expand_dims(img, axis=0)
    return img


def get_distance(embedding1, embedding2):
    """O distance

    Args:
        embedding1: (1, 512)
        embedding2: (num_faces, 512)

    Returns:
        distances

    """
    embedding1 = embedding1 / np.linalg.norm(embedding1)
    embedding2 = embedding2 / np.linalg.norm(embedding2, axis=1)[..., np.newaxis]
    distances = np.sum(np.square(embedding2 - embedding1), axis=-1)

    return distances


def get_similar_face(distances, dis_thres):
    similar_face = np.argmin(distances)
    similar_dis = distances[similar_face]

    if similar_dis < dis_thres:
        return similar_face
    else:
        return []
