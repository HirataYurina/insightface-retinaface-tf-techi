# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:test_net.py
# software: PyCharm

import cv2
from retinaface.net.face import retinaface_net
import tensorflow.keras as keras
import numpy as np

"""
    i have test retinaface network and it is right.
"""

# test the validity of retinaface network
techi = cv2.imread('../img/techi.jpg')
print(techi.shape)
techi = keras.applications.imagenet_utils.preprocess_input(techi)
techi = np.expand_dims(techi, axis=0)

inputs = keras.Input((None, None, 3))
retinaface = retinaface_net(inputs, training=False)
retinaface.load_weights('../model_weights/retinaface_resnet50.h5', by_name=True)
retinaface.summary()

results = retinaface(techi)
print(results)
