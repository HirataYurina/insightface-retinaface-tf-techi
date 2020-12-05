# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:resnet.py
# software: PyCharm


from __future__ import print_function
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.layers import Activation, BatchNormalization
import tensorflow.keras as keras


def identity_block(input_tensor, kernel_size, filters, stage, block, training):
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x, training=training)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x, training=training)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x, training=training)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, training, strides=(2, 2)):
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x, training=training)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x, training=training)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x, training=training)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut, training=training)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(inputs, training=False):
    img_input = inputs
    x = ZeroPadding2D((3, 3))(img_input)

    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x, training=training)
    x = Activation('relu')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), training=training)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', training=training)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', training=training)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', training=training)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', training=training)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', training=training)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', training=training)
    feat1 = x

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', training=training)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', training=training)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', training=training)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', training=training)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', training=training)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', training=training)
    feat2 = x

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', training=training)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', training=training)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', training=training)
    feat3 = x

    return feat1, feat2, feat3


if __name__ == '__main__':

    inputs_ = keras.Input(shape=(416, 416, 3))
    res = ResNet50(inputs_)
    model = keras.Model(inputs_, res)

    model.summary()
