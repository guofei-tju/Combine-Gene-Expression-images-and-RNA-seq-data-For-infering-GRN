# -*- coding: utf-8 -*-
"""
Created on Mon May 11 13:45:20 2020

@author: Administrator
"""
from keras import layers
from keras import Model
import tensorflow as tf


def SDINet(input_shape, nb_classes=1):
    filters = [32, 64, 128, 256]
    input = layers.Input(shape=input_shape)

    def conv1(input):
        con = layers.Conv2D(filters=filters[0], kernel_size=(7, 7), strides=(2, 2), padding='same')(input)
        bn = layers.BatchNormalization()(con)
        ac = layers.Activation('relu')(bn)
        mp = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(ac)
        return mp

    ##########################################################################
    def conv2_x(cat):
        con = layers.Conv2D(filters=filters[0], kernel_size=(3, 3), strides=(1, 1), padding='same')(cat)
        bn = layers.BatchNormalization()(con)
        ac = layers.Activation('relu')(bn)

        con = layers.Conv2D(filters=filters[0], kernel_size=(3, 3), strides=(1, 1), padding='same')(ac)
        bn = layers.BatchNormalization()(con)

        conv2_x_add = layers.add([bn, cat])

        ac = layers.Activation('relu')(conv2_x_add)

        return ac

    def conv3_x(cat, strides=1):
        if strides == 2:
            con = layers.Conv2D(filters=filters[1], kernel_size=(3, 3), strides=(2, 2), padding='same')(cat)  # 特征图尺寸减半，滤波器数量加倍。
        else:
            con = layers.Conv2D(filters=filters[1], kernel_size=(3, 3), strides=(1, 1), padding='same')(cat)  # 相同的输出特征图，层具有相同数量的滤波器

        bn = layers.BatchNormalization()(con)
        ac = layers.Activation('relu')(bn)
        con = layers.Conv2D(filters=filters[1], kernel_size=(3, 3), strides=(1, 1), padding='same')(ac)
        bn1 = layers.BatchNormalization()(con)

        if strides == 2:
            con = layers.Conv2D(filters=filters[1], kernel_size=(3, 3), strides=(2, 2), padding='same')(cat)  # 特征图尺寸减半，滤波器数量加倍。
            bn2 = layers.BatchNormalization()(con)
            conv2_x_add = layers.add([bn1, bn2])
        else:
            conv2_x_add = layers.add([bn1, cat])  # 相同的输出特征图，层具有相同数量的滤波器

        ac = layers.Activation('relu')(conv2_x_add)
        return ac

    def conv4_x(cat, strides=1):
        if strides == 2:
            con = layers.Conv2D(filters=filters[2], kernel_size=(3, 3), strides=(2, 2), padding='same')(cat)  # 特征图尺寸减半，滤波器数量加倍。
        else:
            con = layers.Conv2D(filters=filters[2], kernel_size=(3, 3), strides=(1, 1), padding='same')(cat)  # 相同的输出特征图，层具有相同数量的滤波器

        bn = layers.BatchNormalization()(con)
        ac = layers.Activation('relu')(bn)
        con = layers.Conv2D(filters=filters[2], kernel_size=(3, 3), strides=(1, 1), padding='same')(ac)
        bn1 = layers.BatchNormalization()(con)

        if strides == 2:
            con = layers.Conv2D(filters=filters[2], kernel_size=(3, 3), strides=(2, 2), padding='same')(cat)  # 特征图尺寸减半，滤波器数量加倍。
            bn2 = layers.BatchNormalization()(con)
            conv2_x_add = layers.add([bn1, bn2])
        else:
            conv2_x_add = layers.add([bn1, cat])  # 相同的输出特征图，层具有相同数量的滤波器

        ac = layers.Activation('relu')(conv2_x_add)
        return ac

    def conv5_x(cat, strides=1):
        if strides == 2:
            con = layers.Conv2D(filters=filters[3], kernel_size=(3, 3), strides=(2, 2), padding='same')(cat)  # 特征图尺寸减半，滤波器数量加倍。
        else:
            con = layers.Conv2D(filters=filters[3], kernel_size=(3, 3), strides=(1, 1), padding='same')(cat)  # 相同的输出特征图，层具有相同数量的滤波器

        bn = layers.BatchNormalization()(con)
        ac = layers.Activation('relu')(bn)
        con = layers.Conv2D(filters=filters[3], kernel_size=(3, 3), strides=(1, 1), padding='same')(ac)
        bn1 = layers.BatchNormalization()(con)

        if strides == 2:
            con = layers.Conv2D(filters=filters[3], kernel_size=(3, 3), strides=(2, 2), padding='same')(cat)  # 特征图尺寸减半，滤波器数量加倍。
            bn2 = layers.BatchNormalization()(con)
            conv2_x_add = layers.add([bn1, bn2])
        else:
            if int(bn1._keras_shape[-1]) != int(cat._keras_shape[-1]):
                cat = con = layers.Conv2D(filters=filters[3], kernel_size=(1, 1), padding='same')(cat)  # 特征图尺寸减半，滤波器数量加倍。
                cat = layers.BatchNormalization()(cat)
            conv2_x_add = layers.add([bn1, cat])  # 相同的输出特征图，层具有相同数量的滤波器

        ac = layers.Activation('relu')(conv2_x_add)
        return ac

    def ASPP(input):
        b22 = layers.Conv2D(filters=filters[2], kernel_size=(1, 1), padding='same')(input)
        b22 = layers.BatchNormalization()(b22)
        b22 = layers.Activation('relu')(b22)

        b23 = layers.Conv2D(filters=filters[2], kernel_size=(3, 3), dilation_rate=3, padding='same')(input)
        b23 = layers.BatchNormalization()(b23)
        b23 = layers.Activation('relu')(b23)

        b24 = layers.Conv2D(filters=filters[2], kernel_size=(3, 3), dilation_rate=6, padding='same')(input)
        b24 = layers.BatchNormalization()(b24)
        b24 = layers.Activation('relu')(b24)

        b25 = layers.Conv2D(filters=filters[2], kernel_size=(3, 3), dilation_rate=9, padding='same')(input)
        b25 = layers.BatchNormalization()(b25)
        b25 = layers.Activation('relu')(b25)

        global_pool_b = layers.AveragePooling2D(pool_size=(input.shape[1], input.shape[2]), strides=1)(input)
        # global_pool_b = tf.reduce_mean(input, axis=(-2, -3), keepdims=True)
        b26 = layers.Conv2D(filters=filters[2], kernel_size=(1, 1), padding='same')(global_pool_b)
        b26 = layers.BatchNormalization()(b26)
        b26 = layers.Activation('relu')(b26)
        b26 =layers.UpSampling2D(size=(input.shape[1], input.shape[2]))(b26)

        b27 = layers.concatenate([b22, b23, b24, b25, b26], axis=-1)
        b27 = layers.Conv2D(filters=filters[2], kernel_size=(1, 1), padding='same')(b27)
        b27 = layers.BatchNormalization()(b27)
        b27 = layers.Activation('relu')(b27)
        return b27

    def se_block_var3(input_tensor, c=4):
        num_channels = int(input_tensor._keras_shape[-1])  # Tensorflow backend
        bottleneck = int(num_channels // c)

        se_branch = layers.GlobalAvgPool2D()(input_tensor)
        # se_branch = layers.Dense(bottleneck, use_bias=False, activation='relu')(se_branch)
        se_branch = layers.Dense(num_channels, use_bias=False, activation='sigmoid')(se_branch)
        out = layers.Multiply()([input_tensor, se_branch])
        return out
    ##########################################################################
    con1 = conv1(input)
    ##########################################################################
    con2 = conv2_x(con1)
    con2 = conv2_x(con2)
    ##########################################################################
    con3 = conv3_x(con2, strides=2)
    con3 = conv3_x(con3, strides=1)
    ##########################################################################
    con4 = conv4_x(con3, strides=2)
    con4 = conv4_x(con4, strides=1)
    ##########################################################################
    branch0 = conv5_x(con4, strides=2)
    branch0 = conv5_x(branch0, strides=1)
    branch0 = layers.GlobalAvgPool2D(name='globalavgpool1')(branch0)
    branch1 = ASPP(con2)
    branch1 = se_block_var3(branch1)
    branch1 = conv5_x(branch1, strides=1)
    branch1 = layers.GlobalAvgPool2D(name='globalavgpool2')(branch1)
    con5 = layers.concatenate([branch0, branch1], axis=-1)
    ##########################################################################
    output = layers.Dense(nb_classes, activation='sigmoid', name='dense')(con5)

    model = Model(inputs=[input], outputs=[output])
    return model