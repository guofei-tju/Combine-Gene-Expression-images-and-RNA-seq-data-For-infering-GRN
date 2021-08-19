# -*- coding: utf-8 -*-
"""
Created on Mon May 11 13:45:20 2020

@author: Administrator
"""
from keras import layers
from keras import Model
import tensorflow as tf
import keras


def expression_SDINet(input1_shape, input2_shape, nb_classes=1):
    filters = [32, 64, 128, 256]
    input_1 = layers.Input(shape=input1_shape, name="image_input")
    input_2 = layers.Input(shape=input2_shape, name="expression_input")

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

    def conv2d_bn(input, kernel_num, kernel_size=3, strides=1, padding_mode='same'):
        conv1 = layers.Conv2D(kernel_num, kernel_size, strides=strides, padding=padding_mode)(input)
        batch1 = layers.BatchNormalization()(conv1)
        return batch1
    # ##################################image model#######################################
    con1 = conv1(input_1)

    con2 = conv2_x(con1)                             # 64*80*32
    con2 = conv2_x(con2)

    con3 = conv3_x(con2, strides=2)                  # 32*40*64
    con3 = conv3_x(con3, strides=1)

    con4 = conv4_x(con3, strides=2)                  # 16*20*128
    con4 = conv4_x(con4, strides=1)

    branch0 = conv5_x(con4, strides=2)               # 8*10*256
    branch0 = conv5_x(branch0, strides=1)
    branch0 = layers.GlobalAvgPool2D(name='globalavgpool1')(branch0)
    branch1 = ASPP(con4)                             # 16*20*256
    # branch1 = conv5_x(con4, strides=1)
    branch1 = se_block_var3(branch1)
    branch1 = conv5_x(branch1, strides=1)
    branch1 = layers.GlobalAvgPool2D(name='globalavgpool2')(branch1)
    image_con = layers.concatenate([branch0, branch1], axis=-1)
    # con5 = layers.Dense(128, activation='relu')(con5)
    ##########################################################################
    # #############################表达量网络###################################################
    expression_conv1 = conv2d_bn(input_2, kernel_num=32, kernel_size=3)
    expression_conv1 = layers.Activation('relu')(expression_conv1)
    expression_conv1 = conv2d_bn(expression_conv1, kernel_num=32, kernel_size=3)
    expression_conv1 = layers.Activation('relu')(expression_conv1)
    expression_pool1 = layers.MaxPooling2D(pool_size=(2, 2))(expression_conv1)
    expression_dropout1 = layers.Dropout(0.5)(expression_pool1)

    expression_conv2 = conv2d_bn(expression_dropout1, kernel_num=64, kernel_size=3)
    expression_conv2 = layers.Activation('relu')(expression_conv2)
    expression_conv2 = conv2d_bn(expression_conv2, kernel_num=64, kernel_size=3)
    expression_conv2 = layers.Activation('relu')(expression_conv2)
    expression_pool2 = layers.MaxPooling2D(pool_size=(2, 2))(expression_conv2)
    expression_dropout2 = layers.Dropout(0.5)(expression_pool2)

    expression_conv3 = conv2d_bn(expression_dropout2, kernel_num=128, kernel_size=3)
    expression_conv3 = layers.Activation('relu')(expression_conv3)
    expression_conv3 = conv2d_bn(expression_conv3, kernel_num=128, kernel_size=3)
    expression_conv3 = layers.Activation('relu')(expression_conv3)
    # branch0  # 8*8*256
    expression_pool3 = layers.MaxPooling2D(pool_size=(2, 2))(expression_conv3)
    expression_dropout3 = layers.Dropout(0.5)(expression_pool3)
    expression_conv4 = conv2d_bn(expression_dropout3, kernel_num=256, kernel_size=3)
    expression_conv4 = layers.Activation('relu')(expression_conv4)
    expression_conv4 = conv2d_bn(expression_conv4, kernel_num=256, kernel_size=3)
    expression_conv4 = layers.Activation('relu')(expression_conv4)
    expression_branch0 = layers.GlobalAvgPool2D(name='globalavgpool3')(expression_conv4)
    # branch1 # 16*16*256
    expression_branch1 = ASPP(expression_conv3)
    expression_branch1 = se_block_var3(expression_branch1)
    expression_branch1 = conv2d_bn(expression_branch1, kernel_num=256, kernel_size=3)
    expression_branch1 = layers.Activation('relu')(expression_branch1)
    expression_branch1 = layers.GlobalAvgPool2D(name='globalavgpool4')(expression_branch1)
    expression_con = layers.concatenate([expression_branch0, expression_branch1], axis=-1)
    ##########################################################################
    # #################################融合网络##########################################
    expression_global2 = layers.GlobalAvgPool2D()(expression_conv2)
    image_global2 = layers.GlobalAvgPool2D()(con3)
    joint_con1 = layers.concatenate([expression_global2, image_global2], axis=-1)
    joint_dense1 = layers.Dense(128, activation='relu')(joint_con1)
    expression_global3 = layers.GlobalAvgPool2D()(expression_conv3)
    image_global3 = layers.GlobalAvgPool2D()(con4)
    joint_con2 = layers.concatenate([expression_global3, image_global3, joint_dense1], axis=-1)
    joint_dense2 = layers.Dense(256, activation='relu')(joint_con2)
    # ########################final layer#############################################
    final_conv = layers.concatenate([image_con, joint_dense2, expression_con], axis=-1)
    final_conv = layers.Dense(128, activation='relu', name='dense')(final_conv)
    output = layers.Dense(nb_classes, activation='sigmoid', name='final_dense')(final_conv)

    model = Model(inputs=[input_1, input_2], outputs=output)

    return model