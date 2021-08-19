from __future__ import absolute_import
from __future__ import print_function
# from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
# from keras.layers.advanced_activations import PReLU
# from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.optimizers import adam, Adadelta, adagrad, SGD
# from keras.utils import np_utils, generic_utils
# from six.moves import range
from integration import acc_and_f1
import os
import keras
from sklearn import metrics
# import csv
# from keras import applications
# from keras.models import Model
# from keras.layers.normalization import BatchNormalization
# from keras.metrics import binary_accuracy
# from keras.preprocessing.image import ImageDataGenerator
# from keras.callbacks import EarlyStopping
# from keras.utils import to_categorical
# from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras.utils import multi_gpu_model

from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, \
    Conv2DTranspose, concatenate, GlobalAvgPool2D, MaxPool2D
from keras.layers import add, Flatten, Activation, Add, ReLU
from keras.regularizers import l2
from keras import layers
import warnings

from keras_applications import get_submodules_from_kwargs
from keras_applications import imagenet_utils
from keras_applications.imagenet_utils import decode_predictions
from keras_applications.imagenet_utils import _obtain_input_shape
import keras.backend as K
import numpy as np
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
from keras.models import load_model

import xlwt


# 写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('accuracy'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_accuracy'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('accuracy'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_accuracy'))

    def loss_plot(self, loss_type, round):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train accuracy')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val accuracy')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('accuracy-loss')
        plt.legend(loc="upper right")
        plt.savefig("loss_acc" + str(round) + ".png")


# 将数据写入新文件
def data_write(file_path, datas):
    f = xlwt.Workbook(encoding='utf-8')
    sheet1 = f.add_sheet(u'predict', cell_overwrite_ok=True)  # 创建sheet

    # 将数据写入第 i 行，第 j 列
    i = 0
    for data in datas:
        sheet1.write(i, 0, data[0])
        sheet1.write(i, 1, data[1])
        sheet1.write(i, 2, data[2])
        sheet1.write(i, 3, data[3])
        i = i + 1

    f.save(file_path)  # 保存文件


def expression_model(round, train_data, train_label, valid_data, valid_label, test_data, test_label):
    def conv2d_bn(input, kernel_num, kernel_size=3, strides=1, padding_mode='same'):
        conv1 = Conv2D(kernel_num, kernel_size, strides=strides, padding=padding_mode)(input)
        batch1 = BatchNormalization()(conv1)
        return batch1

    def ASPP(input):
        b22 = layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same')(input)
        b22 = layers.BatchNormalization()(b22)
        b22 = layers.Activation('relu')(b22)

        b23 = layers.Conv2D(filters=128, kernel_size=(3, 3), dilation_rate=3, padding='same')(input)
        b23 = layers.BatchNormalization()(b23)
        b23 = layers.Activation('relu')(b23)

        b24 = layers.Conv2D(filters=128, kernel_size=(3, 3), dilation_rate=6, padding='same')(input)
        b24 = layers.BatchNormalization()(b24)
        b24 = layers.Activation('relu')(b24)

        b25 = layers.Conv2D(filters=128, kernel_size=(3, 3), dilation_rate=9, padding='same')(input)
        b25 = layers.BatchNormalization()(b25)
        b25 = layers.Activation('relu')(b25)

        global_pool_b = layers.AveragePooling2D(pool_size=(input.shape[1], input.shape[2]), strides=1)(input)
        # global_pool_b = tf.reduce_mean(input, axis=(-2, -3), keepdims=True)
        b26 = layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same')(global_pool_b)
        b26 = layers.BatchNormalization()(b26)
        b26 = layers.Activation('relu')(b26)
        b26 = layers.UpSampling2D(size=(input.shape[1], input.shape[2]))(b26)

        b27 = layers.concatenate([b22, b23, b24, b25, b26], axis=-1)
        b27 = layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same')(b27)
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

    filters = [32, 64, 128, 256]

    def expression_ASPP(input):
        b22 = layers.Conv2D(filters=filters[2], kernel_size=(1, 1), padding='same')(input)
        b22 = layers.BatchNormalization()(b22)
        b22 = layers.Activation('relu')(b22)

        b23 = layers.Conv2D(filters=filters[2], kernel_size=(3, 3), dilation_rate=2, padding='same')(input)
        b23 = layers.BatchNormalization()(b23)
        b23 = layers.Activation('relu')(b23)

        b24 = layers.Conv2D(filters=filters[2], kernel_size=(3, 3), dilation_rate=3, padding='same')(input)
        b24 = layers.BatchNormalization()(b24)
        b24 = layers.Activation('relu')(b24)

        b25 = layers.Conv2D(filters=filters[2], kernel_size=(3, 3), dilation_rate=5, padding='same')(input)
        b25 = layers.BatchNormalization()(b25)
        b25 = layers.Activation('relu')(b25)

        global_pool_b = layers.AveragePooling2D(pool_size=(input.shape[1], input.shape[2]), strides=1)(input)
        # global_pool_b = tf.reduce_mean(input, axis=(-2, -3), keepdims=True)
        b26 = layers.Conv2D(filters=filters[2], kernel_size=(1, 1), padding='same')(global_pool_b)
        b26 = layers.BatchNormalization()(b26)
        b26 = layers.Activation('relu')(b26)
        b26 = layers.UpSampling2D(size=(input.shape[1], input.shape[2]))(b26)

        b27 = layers.concatenate([b22, b23, b24, b25, b26], axis=-1)
        b27 = layers.Conv2D(filters=filters[2], kernel_size=(1, 1), padding='same')(b27)
        b27 = layers.BatchNormalization()(b27)
        b27 = layers.Activation('relu')(b27)
        return b27

    input_2 = layers.Input(shape=train_data.shape[1:], name="expression_input")
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
    branch0 = layers.Activation('relu')(expression_conv4)
    branch0 = layers.GlobalAvgPool2D(name='globalavgpool1')(branch0)
    # # branch1 # 16*16*256
    branch1 = ASPP(expression_conv3)
    branch1 = se_block_var3(branch1)
    branch1 = conv2d_bn(branch1, kernel_num=256, kernel_size=3)
    branch1 = layers.Activation('relu')(branch1)
    branch1 = layers.GlobalAvgPool2D(name='globalavgpool2')(branch1)
    # final layer
    con5 = layers.concatenate([branch0, branch1], axis=-1)
    output = layers.Dense(1, activation='sigmoid', name='dense')(con5)

    model = Model(inputs=[input_2], outputs=[output])

    def loss(y_true, y_pred):
        return keras.losses.binary_crossentropy(y_true, y_pred, label_smoothing=0.2)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"])
    with strategy.scope():
        gpu_devices = tf.config.experimental.list_physical_devices('gpu')
        if gpu_devices:
            for gpu in gpu_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
    parallel_model = model
    parallel_model.compile(loss='binary_crossentropy', optimizer=adam(lr=3e-4), metrics=['accuracy'])
    parallel_model.summary()
    nb_epochs = 300

    # # 创建一个实例history
    history = LossHistory()
    tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    save_best_model = ModelCheckpoint('./output/Vgg_expression_model(64)_best_model' + str(round) + '.h5',
                                      monitor='val_accuracy',
                                      save_best_only=True)
    save_last_model = ModelCheckpoint('./output/Vgg_expression_model(64)_last_model' + str(round) + '.h5',
                                      monitor='val_accuracy',
                                      save_best_only=False)

    # 学习率下降
    def step_decay(epoch):
        if epoch == 40:
            K.set_value(parallel_model.optimizer.lr, K.get_value(parallel_model.optimizer.lr) * 0.1)
        if epoch == 70:
            K.set_value(parallel_model.optimizer.lr, K.get_value(parallel_model.optimizer.lr) * 0.1)
        if epoch == 90:
            K.set_value(parallel_model.optimizer.lr, K.get_value(parallel_model.optimizer.lr) * 0.1)
        print("学习率为:", K.get_value(parallel_model.optimizer.lr))
        return K.get_value(parallel_model.optimizer.lr)

    print_lrate = keras.callbacks.LearningRateScheduler(step_decay)

    hist = parallel_model.fit(train_data, train_label, batch_size=32, epochs=nb_epochs, shuffle=True, verbose=1,
                              validation_data=(valid_data, valid_label),
                              callbacks=[save_best_model, save_last_model, tbCallBack, history])

    history.loss_plot('epoch', round)

    # model = load_model('./output/Unet-transferlearning2-mesoderm.h5')

    test_predict = parallel_model.predict(test_data)

    test_label = np.concatenate((test_label, np.array(test_predict)), axis=-1)
    data_write('./output/expression_prediction' + str(round) + '.xls', test_label)
    label = []
    for j in test_label:
        label.append([float(j[2])])
    test_label = np.array(label)

    acc1 = metrics.accuracy_score(test_label, (test_predict >= 0.5).astype(int))
    f1 = metrics.f1_score(test_label, (test_predict >= 0.5).astype(int))

    fpr, tpr, thresholds = metrics.roc_curve(test_label, test_predict)
    AUC_ROC = metrics.roc_auc_score(test_label, test_predict)

    # 画AUC曲线
    plt.figure()
    plt.plot(fpr, tpr, 'r--', label='expression_model (AUC = %0.4f)' % AUC_ROC)
    plt.title('ROC curve')
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.legend(loc="lower right")
    plt.savefig('./output/AUC' + str(round) + '.png')

    precision, recall, _thresholds = metrics.precision_recall_curve(test_label, test_predict)
    aupr = metrics.auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, 'r--', label='expression_model (AUPR = %0.4f)' % aupr)
    plt.title('PR curve')
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="lower right")
    plt.savefig('./output/AUPR' + str(round) + '.png')

    confusion_matrix = metrics.confusion_matrix(test_label, (test_predict >= 0.5).astype(int))
    TN = confusion_matrix[0][0]
    FN = confusion_matrix[0][1]
    FP = confusion_matrix[1][0]
    TP = confusion_matrix[1][1]

    print("TN,TP,FN,FP:", TN, TP, FN, FP)
    print("acc:", acc1)
    print("f1:", f1)
    print("Auc:", AUC_ROC)
    print("Aupr:", aupr)
    return acc1, f1, AUC_ROC, aupr, TN, TP, FN, FP
