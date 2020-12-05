# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:layers.py
# software: PyCharm

import tensorflow.keras as keras
import tensorflow as tf

backend = tf


class UpsampleLike(keras.layers.Layer):
    """upsample like
       upsample like can resize tensor to any shape
       (n // 4 * 2) != (n // 2) so i use upsample like
    """
    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = keras.backend.shape(target)
        if keras.backend.image_data_format() == 'channels_first':
            source = backend.transpose(source, (0, 2, 3, 1))
            output = tf.compat.v1.image.resize_images(source, (target_shape[2], target_shape[3]), method='nearest')
            output = backend.transpose(output, (0, 3, 1, 2))
            return output
        else:
            return tf.compat.v1.image.resize_images(source, (target_shape[1], target_shape[2]), method='nearest')

    def compute_output_shape(self, input_shape):
        if keras.backend.image_data_format() == 'channels_first':
            return (input_shape[0][0], input_shape[0][1]) + input_shape[1][2:4]
        else:
            return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)
