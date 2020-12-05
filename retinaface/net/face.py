# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:retinaface.py
# software: PyCharm

from retinaface.net.fpn import fpn, cov_block
from retinaface.net.resnet import ResNet50
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


def cov_block_no_leaky(inputs, out_channel, kernel_size, strides, padding, name=None, bn_name=None,
                       training=False):
    x = layers.Conv2D(out_channel, kernel_size, strides, padding=padding, name=name)(inputs)
    x = layers.BatchNormalization(name=bn_name)(x, training=training)
    # x = layers.LeakyReLU(alpha=leaky_alpha)(x)

    return x


def context_module(inputs, out_channel, kernel_size, strides, padding, leaky_alpha, name=None, bn_name=None,
                   training=False):
    """use context module to enhance respective field"""

    x1 = cov_block_no_leaky(inputs, out_channel // 2, kernel_size, strides, padding,
                            name='conv2d_' + str(name),
                            bn_name='batch_normalization_' + str(bn_name),
                            training=training)

    x2_1 = cov_block(inputs, out_channel // 4, kernel_size, strides, padding, leaky_alpha=leaky_alpha,
                     name='conv2d_' + str(name + 1),
                     bn_name='batch_normalization_' + str(bn_name + 1),
                     training=training)
    x2_2 = cov_block_no_leaky(x2_1, out_channel // 4, kernel_size, strides, padding,
                              name='conv2d_' + str(name + 2),
                              bn_name='batch_normalization_' + str(bn_name + 2),
                              training=training)

    x3_1 = cov_block(x2_1, out_channel // 4, kernel_size, strides, padding, leaky_alpha=leaky_alpha,
                     name='conv2d_' + str(name + 3),
                     bn_name='batch_normalization_' + str(bn_name + 3),
                     training=training)
    x3_2 = cov_block_no_leaky(x3_1, out_channel // 4, kernel_size, strides, padding,
                              name='conv2d_' + str(name + 4),
                              bn_name='batch_normalization_' + str(bn_name + 4),
                              training=training)

    merge = layers.Concatenate()([x1, x2_2, x3_2])
    merge = layers.ReLU()(merge)

    return merge


def classification(inputs, num_anchors=2, name=None):
    x = layers.Conv2D(num_anchors * 2, 1, padding='same', name='conv2d_' + str(name))(inputs)
    x = layers.Reshape(target_shape=(-1, 2))(x)  # (52*52*num_anchors, 2)
    x = layers.Softmax()(x)

    return x


def bbox(inputs, num_anchors=2, name=None):
    x = layers.Conv2D(num_anchors * 4, 1, padding='same', name='conv2d_' + str(name))(inputs)
    x = layers.Reshape((-1, 4))(x)

    return x


def landmarks(inputs, num_anchors=2, name=None):
    x = layers.Conv2D(num_anchors * 10, 1, padding='same', name='conv2d_' + str(name))(inputs)
    x = layers.Reshape((-1, 10))(x)

    return x


def retinaface_net(inputs, training=False):
    """retinaface network"""
    c3, c4, c5 = ResNet50(inputs, training)

    out_channel = 256
    leaky_alpha = 0.0
    p3, p4, p5 = fpn(c3, c4, c5, out_channel, leaky_alpha, training)

    # use context module to enhance respective field
    p3 = context_module(p3, out_channel, 3, 1, 'same', leaky_alpha=leaky_alpha,
                        name=1, bn_name=6, training=training)
    p4 = context_module(p4, out_channel, 3, 1, 'same', leaky_alpha=leaky_alpha,
                        name=6, bn_name=11, training=training)
    p5 = context_module(p5, out_channel, 3, 1, 'same', leaky_alpha=leaky_alpha,
                        name=11, bn_name=16, training=training)

    # prediction
    # classification prediction
    classi = layers.Concatenate(axis=1)([classification(x, name=19 + i) for i, x in enumerate([p3, p4, p5])])

    # regression prediction
    regression = layers.Concatenate(axis=1)([bbox(x, name=16 + i) for i, x in enumerate([p3, p4, p5])])

    # landmarks prediction
    ld = layers.Concatenate(axis=1)([landmarks(x, name=22 + i) for i, x in enumerate([p3, p4, p5])])

    return keras.Model(inputs, [regression, classi, ld])


if __name__ == '__main__':
    inputs_ = keras.Input(shape=(416, 416, 3))
    model = retinaface_net(inputs_)
    model.summary()
    # print(len(model.layers))
    print('starting loading weights')
    model.load_weights('../model_weights/retinaface_resnet50.h5', by_name=True)
    print('loading successfully!')
