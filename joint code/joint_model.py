from __future__ import absolute_import
from __future__ import print_function

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.optimizers import adam, Adadelta, adagrad, SGD

from integration import acc_and_f1
import os
import keras
import csv

from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras.utils import multi_gpu_model
from sklearn import metrics
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, \
    Conv2DTranspose, concatenate, GlobalAvgPool2D, MaxPool2D
from keras.layers import add, Flatten, Activation, Add, ReLU, Multiply
from keras.regularizers import l2
import warnings
from keras.utils import plot_model
from joint_model1 import expression_SDINet
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
from keras import layers


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


def my_model(round, train_data, train_label, valid_data, valid_label, test_data, test_label,
                   train_expression_data, train_expression_label,
                   valid_expression_data, valid_expression_label,
                   test_expression_data, test_expression_label):

    model = expression_SDINet(input1_shape=train_data.shape[1:], input2_shape=train_expression_data.shape[1:])

    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
    with strategy.scope():
        gpu_devices = tf.config.experimental.list_physical_devices('gpu')
        if gpu_devices:
            for gpu in gpu_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
        parallel_model = multi_gpu_model(model, gpus=2)

    # parallel_model = model

    def loss(y_true, y_pred):
        return keras.losses.binary_crossentropy(y_true, y_pred, label_smoothing=0.2)

    parallel_model.compile(loss='binary_crossentropy', optimizer=adam(lr=3e-4),
                           metrics=['accuracy'])  # 'binary_crossentropy'
    parallel_model.summary()

    nb_epochs = 60

    # # 创建一个实例history
    history = LossHistory()
    tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    save_best_model = ModelCheckpoint('./output/joint_model2_RBF_best_model' + str(round) + '.h5',
                                      monitor='val_accuracy',
                                      save_best_only=True)
    save_last_model = ModelCheckpoint('./output/joint_model2_RBF_last_model' + str(round) + '.h5',
                                      monitor='val_accuracy',
                                      save_best_only=False)

    # 学习率下降
    def step_decay(epoch):
        if epoch == 10:
            K.set_value(parallel_model.optimizer.lr, K.get_value(parallel_model.optimizer.lr) * 0.1)
        if epoch == 15:
            K.set_value(parallel_model.optimizer.lr, K.get_value(parallel_model.optimizer.lr) * 0.1)
        # if epoch == 50:
        #     K.set_value(parallel_model.optimizer.lr, K.get_value(parallel_model.optimizer.lr) * 0.1)
        print("学习率为:", K.get_value(parallel_model.optimizer.lr))
        return K.get_value(parallel_model.optimizer.lr)

    print_lrate = keras.callbacks.LearningRateScheduler(step_decay)

    hist = parallel_model.fit({"image_input": train_data, "expression_input": train_expression_data}, train_label,
                              batch_size=32, epochs=nb_epochs, shuffle=False, verbose=1,
                              validation_data=(
                              {"image_input": valid_data, "expression_input": valid_expression_data}, valid_label),
                              callbacks=[save_best_model, save_last_model, history])

    history.loss_plot('epoch', round)

    # #############################################预测结果#############################################################

    test_predict = parallel_model.predict({"image_input": test_data, "expression_input": test_expression_data})
    predict_list = []

    with open("output/joint_model2_prediction_mesoderm-round" + str(round) + ".csv", 'w') as f:
        for i, row in enumerate(test_label):
            f.write(str(row[0]) + ',' + str(row[1]) + ',' + str(row[2]) + ',' + str(test_predict[i][0]) + '\n')
            new_row = [row[0], row[1], row[2], test_predict[i][0]]
            predict_list.append(new_row)
    f.close()

    acc1, f1, gene_pairs, FP, TP, FN, TN = acc_and_f1(predict_list)
    # print('test acc: ', acc1, 'test f1: ', f1)

    with open("output/joint_model2_mesoderm_prediction_integration-round" + str(round) + ".csv", 'w') as f:
        for edge in gene_pairs.edges:
            f.write(str(edge.geneA) + ',' + str(edge.geneB) + ',' + str(edge.groundTruth) + ',' + str(
                edge.final_pred) + '\n')
    f.close()

    test_label = []
    test_predict = []
    with open("output/joint_model2_mesoderm_prediction_integration-round" + str(round) + ".csv", 'r') as f:
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
    plt.plot(fpr, tpr, 'r--', label='expression_model (AUC = %0.4f)' % AUC_ROC)
    plt.title('ROC curve')
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.legend(loc="lower right")
    plt.savefig('./output/AUC.png')

    precision, recall, _thresholds = metrics.precision_recall_curve(test_label, test_predict)
    aupr = metrics.auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, 'r--', label='expression_model (AUPR = %0.4f)' % aupr)
    plt.title('PR curve')
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="lower right")
    plt.savefig('./output/AUPR.png')

    print("TN,TP,FN,FP:", TN, TP, FN, FP)
    print("acc:", acc1)
    print("f1:", f1)
    print("Auc:", AUC_ROC)
    print("Aupr:", aupr)

    return acc1, f1, AUC_ROC, aupr, TN, TP, FN, FP
