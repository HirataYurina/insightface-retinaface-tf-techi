# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:fpn.py
# software: PyCharm

import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from retinaface.net.resnet import ResNet50
from retinaface.net.layers import UpsampleLike


def cov_block(inputs, out_channel, kernel_size, strides, padding, training, name=None, leaky_alpha=0.1, bn_name=None):
    x = layers.Conv2D(out_channel, kernel_size, strides, padding=padding, name=name)(inputs)
    x = layers.BatchNormalization(name=bn_name)(x, training=training)
    x = layers.LeakyReLU(alpha=leaky_alpha)(x)

    return x


def fpn(c3, c4, c5, out_channel, leaky_alpha, training=False):
    """feature pyramid network

    Args:
        c3: n/8
        c4: n/16
        c5: n/32
        out_channel
        leaky_alpha
        training:   training or not training is different for BN layer

    Returns:
        p3: n/8
        p4: n/16
        p5: n/32

    """
    p3 = cov_block(c3, out_channel, 1, 1, 'same', training, 'C3_reduced', leaky_alpha, bn_name='batch_normalization_1')
    p4 = cov_block(c4, out_channel, 1, 1, 'same', training, 'C4_reduced', leaky_alpha, bn_name='batch_normalization_2')
    p5 = cov_block(c5, out_channel, 1, 1, 'same', training, 'C5_reduced', leaky_alpha, bn_name='batch_normalization_3')

    # p5 up sample
    p5_up = UpsampleLike(name='P5_upsampled')([p5, p4])
    p4 = layers.Add(name='P4_merged')([p5_up, p4])
    p4 = cov_block(p4, out_channel, 3, 1, 'same', training, 'Conv_P4_merged', leaky_alpha,
                   bn_name='batch_normalization_4')

    # p4 up sample
    p4_up = UpsampleLike(name='P4_upsampled')([p4, p3])
    p3 = layers.Add(name='P3_merged')([p4_up, p3])
    p3 = cov_block(p3, out_channel, 3, 1, 'same', training, 'Conv_P3_merged', leaky_alpha,
                   bn_name='batch_normalization_5')

    return p3, p4, p5


if __name__ == '__main__':
    inputs_ = keras.Input(shape=(416, 416, 3))
    c3_, c4_, c5_ = ResNet50(inputs_)
    p3_, p4_, p5_ = fpn(c3_, c4_, c5_, 256, 0.1)

    model = keras.Model(inputs_, [p3_, p4_, p5_])
    model.summary()
