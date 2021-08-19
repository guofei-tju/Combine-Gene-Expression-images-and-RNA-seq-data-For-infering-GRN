from __future__ import absolute_import
from __future__ import print_function

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.optimizers import adam, Adadelta, adagrad, SGD

from integration import acc_and_f1
import os
import keras
import csv
from resnext import ResNext, ResNextImageNet
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import tensorflow as tf
from keras.utils import multi_gpu_model
from sklearn import metrics
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, \
    Conv2DTranspose, concatenate, GlobalAvgPool2D, MaxPool2D
from keras.layers import add, Flatten, Activation, Add, ReLU, Multiply
from keras.regularizers import l2
import warnings
from inception_resnet_v2 import InceptionResNetV2
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


def my_model(round, train_data, train_label, valid_data, valid_label, test_data, test_label):
    def conv2d_bn(input, kernel_num, kernel_size=3, strides=1, layer_name='', padding_mode='same'):
        conv1 = Conv2D(kernel_num, kernel_size, strides=strides, padding=padding_mode, name=layer_name + '_conv1')(
            input)
        batch1 = BatchNormalization(name=layer_name + '_bn1')(conv1)
        return batch1

    def shortcut(fx, x, padding_mode='same', layer_name='', downsample_flag=0):
        if downsample_flag == 0:
            layer_name += '_shortcut'
            if x.shape[-1] != fx.shape[-1]:
                k = fx.shape[-1]
                k = int(k)
                identity = conv2d_bn(x, kernel_num=k, kernel_size=1, padding_mode=padding_mode, layer_name=layer_name)
            else:
                identity = x
        else:  # 下采样
            layer_name += '_shortcut'
            if x.shape[-1] != fx.shape[-1]:
                k = fx.shape[-1]
                k = int(k)
                pool = AveragePooling2D(pool_size=[3, 3], strides=2, padding='same')(x)
                identity = conv2d_bn(pool, kernel_num=k, kernel_size=1, padding_mode=padding_mode,
                                     layer_name=layer_name)
            else:
                identity = AveragePooling2D(pool_size=[3, 3], strides=2, padding='same')(x)

        return Add(name=layer_name + '_add')([identity, fx])

    def bottleneck(input, kernel_num, strides=1, layer_name='bottleneck', padding_mode='same'):
        k1, k2, k3 = kernel_num
        conv1 = conv2d_bn(input, kernel_num=k1, kernel_size=1, strides=strides, padding_mode=padding_mode,
                          layer_name=layer_name + '_1')
        relu1 = ReLU(name=layer_name + '_relu1')(conv1)
        conv2 = conv2d_bn(relu1, kernel_num=k2, kernel_size=3, strides=strides, padding_mode=padding_mode,
                          layer_name=layer_name + '_2')
        relu2 = ReLU(name=layer_name + '_relu2')(conv2)
        conv3 = conv2d_bn(relu2, kernel_num=k3, kernel_size=1, strides=strides, padding_mode=padding_mode,
                          layer_name=layer_name + '_3')
        # print(conv3.shape, input.shape)
        shortcut_add = shortcut(fx=conv3, x=input, layer_name=layer_name)
        relu3 = ReLU(name=layer_name + '_relu3')(shortcut_add)

        return relu3

    def basic_block(input, kernel_num=64, strides=1, layer_name='basic', padding_mode='same', downsample_flag=0):
        # k1, k2 = kernel
        if downsample_flag == 0:
            conv1 = conv2d_bn(input, kernel_num=kernel_num, strides=strides, kernel_size=3,
                              layer_name=layer_name + '_1', padding_mode=padding_mode)
            relu1 = ReLU(name=layer_name + '_relu1')(conv1)
            conv2 = conv2d_bn(relu1, kernel_num=kernel_num, strides=strides, kernel_size=3,
                              layer_name=layer_name + '_2', padding_mode=padding_mode)
            relu2 = ReLU(name=layer_name + '_relu2')(conv2)
            # relu2 = se_block(relu2, c=16)
            shortcut_add = shortcut(fx=relu2, x=input, layer_name=layer_name, downsample_flag=downsample_flag)
            relu3 = ReLU(name=layer_name + '_relu3')(shortcut_add)
        else:
            conv1 = conv2d_bn(input, kernel_num=kernel_num, strides=1, kernel_size=3,
                              layer_name=layer_name + '_1', padding_mode=padding_mode)
            relu1 = ReLU(name=layer_name + '_relu1')(conv1)
            conv2 = conv2d_bn(relu1, kernel_num=kernel_num, strides=2, kernel_size=3,
                              layer_name=layer_name + '_2', padding_mode=padding_mode)
            relu2 = ReLU(name=layer_name + '_relu2')(conv2)
            # relu2 = se_block(relu2, c=16)
            shortcut_add = shortcut(fx=relu2, x=input, layer_name=layer_name, downsample_flag=downsample_flag)
            relu3 = ReLU(name=layer_name + '_relu3')(shortcut_add)
        return relu3

    def make_layer(input, block, block_num, kernel_num, layer_name=''):
        x = input
        for i in range(1, block_num):
            x = block(x, kernel_num=kernel_num, strides=1, layer_name=layer_name + str(i), padding_mode='same',
                      downsample_flag=0)
        x = block(x, kernel_num=kernel_num, strides=1, layer_name=layer_name + str(block_num), padding_mode='same',
                  downsample_flag=1)
        return x

    def se_block(input_tensor, c=16):
        num_channels = int(input_tensor._keras_shape[-1])  # Tensorflow backend
        bottleneck = int(num_channels // c)

        se_branch = GlobalAvgPool2D()(input_tensor)
        se_branch = Dense(bottleneck, use_bias=False, activation='relu')(se_branch)
        se_branch = Dropout(0.2)(se_branch)
        se_branch = Dense(num_channels, use_bias=False, activation='sigmoid')(se_branch)
        se_branch = Dropout(0.2)(se_branch)

        out = Multiply()([input_tensor, se_branch])
        return out

    def ResNet(input_shape, nclass, net_name='resnet18'):
        """
            :param input_shape:
            :param nclass:
            :param block:
            :return:
        """
        block_setting = {}
        block_setting['resnet18'] = {'block': basic_block, 'block_num': [2, 2, 2, 2], 'kernel_num': [32, 64, 128, 256]}
        block_setting['resnet34'] = {'block': basic_block, 'block_num': [3, 4, 6, 3], 'kernel_num': [64, 128, 256, 512]}
        block_setting['resnet50'] = {'block': bottleneck, 'block_num': [3, 4, 6, 3],
                                     'kernel_num': [[64, 64, 256], [128, 128, 512],
                                                    [256, 256, 1024], [512, 512, 2048]]}
        block_setting['resnet101'] = {'block': bottleneck, 'block_num': [3, 4, 23, 3],
                                      'kernel_num': [[64, 64, 256], [128, 128, 512],
                                                     [256, 256, 1024], [512, 512, 2048]]}
        block_setting['resnet152'] = {'block': bottleneck, 'block_num': [3, 8, 36, 3],
                                      'kernel_num': [[64, 64, 256], [128, 128, 512],
                                                     [256, 256, 1024], [512, 512, 2048]]}
        net_name = 'resnet18' if not block_setting.__contains__(net_name) else net_name
        block_num = block_setting[net_name]['block_num']
        kernel_num = block_setting[net_name]['kernel_num']
        block = block_setting[net_name]['block']

        input_ = Input(shape=input_shape)
        conv1 = conv2d_bn(input_, 64, kernel_size=7, strides=2, layer_name='first_conv')
        pool1 = MaxPool2D(pool_size=(3, 3), strides=2, padding='same', name='pool1')(conv1)

        conv = pool1
        for i in range(4):
            conv = make_layer(conv, block=block, block_num=block_num[i], kernel_num=kernel_num[i],
                              layer_name='layer' + str(i + 1))

        pool2 = GlobalAvgPool2D(name='globalavgpool')(conv)
        output_ = Dense(nclass, activation='sigmoid', name='dense')(pool2)

        model = Model(inputs=input_, outputs=output_, name='ResNet18')

        return model

    from my_model import SDINet
    model = SDINet(input_shape=train_data.shape[1:], nb_classes=1)
    # model = ResNet(train_data.shape[1:], 1, net_name='resnet18')
    # model = my_model1(input_shape1=tf_train_data.shape[1:], input_shape2=gene_train_data.shape[1:], classes=1)
    # model = ResNet50(include_top=True, input_shape=train_data.shape[1:], classes=1)
    # model = ResNet(train_data.shape[1:], 1, net_name='resnet18')
    # model = ResNext(input_shape=train_data.shape[1:], depth=29, cardinality=32, width=4, classes=1)
    # model = ResNextImageNet(input_shape=train_data.shape[1:], classes=1)
    # model = InceptionResNetV2(input_shape=train_data.shape[1:], classes=1)

    os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:2", "/gpu:3"])
    with strategy.scope():
        gpu_devices = tf.config.experimental.list_physical_devices('gpu')
        if gpu_devices:
            for gpu in gpu_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
        parallel_model = multi_gpu_model(model, gpus=2)
    # parallel_model = model
    parallel_model.compile(loss='binary_crossentropy', optimizer=adam(lr=3e-6), metrics=['accuracy'])
    parallel_model.summary()
    nb_epochs = 40

    # # 创建一个实例history
    history = LossHistory()
    tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    save_best_model = ModelCheckpoint('./output/256_320_resnext29_image_best_model' + str(round) + '.h5',
                                      monitor='val_accuracy', save_best_only=True)
    save_last_model = ModelCheckpoint('./output/256_320_resnext29_image_last_model' + str(round) + '.h5',
                                      monitor='val_accuracy', save_best_only=False)

    # 学习率下降
    def step_decay(epoch):
        # if epoch==0:
        #     initial_lrate = 1.0
        #     K.set_value(model.optimizer.lr, initial_lrate * 0.1)
        print("学习率为:", K.get_value(parallel_model.optimizer.lr))
        return K.get_value(parallel_model.optimizer.lr)

    print_lrate = LearningRateScheduler(step_decay)

    hist = parallel_model.fit(train_data, train_label, batch_size=16, epochs=nb_epochs, shuffle=True, verbose=1,
                              validation_data=(valid_data, valid_label),
                              callbacks=[save_best_model, save_last_model, tbCallBack, history, print_lrate])

    history.loss_plot('epoch', round)

    ##############################################预测结果################################################################

    test_predict = parallel_model.predict(test_data)
    predict_list = []

    with open("output/256_320_resnext29_prediction_mesoderm-round" + str(round) + ".csv", 'w') as f:
        for i, row in enumerate(test_label):
            f.write(str(row[0]) + ',' + str(row[1]) + ',' + str(row[2]) + ',' + str(test_predict[i][0]) + '\n')
            new_row = [row[0].split('/')[-3], row[1].split('/')[-3], row[2], test_predict[i][0]]
            predict_list.append(new_row)
    f.close()

    acc1, f1, gene_pairs, FP, TP, FN, TN = acc_and_f1(predict_list)
    # print('test acc: ', acc1, 'test f1: ', f1)

    with open("output/256_320_resnext29_mesoderm_prediction_integration-round" + str(round) + ".csv", 'w') as f:
        for edge in gene_pairs.edges:
            f.write(str(edge.geneA) + ',' + str(edge.geneB) + ',' + str(edge.groundTruth) + ',' + str(
                edge.final_pred) + '\n')
    f.close()

    test_label = []
    test_predict = []
    with open("output/256_320_resnext29_mesoderm_prediction_integration-round" + str(round) + ".csv", 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            test_label.append(float(row[2]))
            test_predict.append(float(row[3]))
    f.close()
    test_label = np.array(test_label)
    test_predict = np.array(test_predict)

    fpr, tpr, thresholds = metrics.roc_curve(test_label, test_predict)
    AUC_ROC = metrics.roc_auc_score(test_label, test_predict)

    # 画AUC曲线
    plt.figure()
    plt.plot(fpr, tpr, 'r--', label='image_model (AUC = %0.4f)' % AUC_ROC)
    plt.title('ROC curve')
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.legend(loc="lower right")
    plt.savefig('./output/AUC' + str(round) + '.png')

    precision, recall, _thresholds = metrics.precision_recall_curve(test_label, test_predict)
    aupr = metrics.auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, 'r--', label='image_model (AUPR = %0.4f)' % aupr)
    plt.title('PR curve')
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="lower right")
    plt.savefig('./output/AUPR' + str(round) + '.png')

    print("TN,TP,FN,FP:", TN, TP, FN, FP)
    print("acc:", acc1)
    print("f1:", f1)
    print("Auc:", AUC_ROC)
    print("Aupr:", aupr)

    return acc1, f1, AUC_ROC, aupr, TN, TP, FN, FP
