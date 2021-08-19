import os
import csv
import numpy as np
from PIL import Image
import random
import xlwt
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import cv2
import xlrd

tf_path = 'tf_dataset.xlsx'
gene_path = 'gene_dataset.xlsx'
database_path = './image_dataset/'
gene_len = 1233
tf_len = 96


def kernel_RBF(X, Y, gamma=0.01):
    r2 = np.repeat(np.sum(X * X, 1), len(X)).reshape(len(X), -1) + \
         np.repeat(np.sum(Y * Y, 1), len(Y)).reshape(len(Y), -1).T - \
         2 * np.dot(X, Y.T)
    return np.exp(-r2 * gamma)


def read_image(img_url):
    # arr1 = np.zeros((128,320,1),dtype="float64")
    img = Image.open(img_url).convert('L')
    # img = Image.open(img_url)
    arr1 = np.asarray(img, "float64")
    '''arr_out[0, :, :] = arr1[:, :, 0]
    arr_out[1, :, :] = arr1[:, :, 1]
    arr_out[2, :, :] = arr1[:, :, 2]'''
    return arr1


# 将数据写入新文件
def data_write(file_path, datas):
    f = xlwt.Workbook(encoding='utf-8')
    sheet1 = f.add_sheet(u'tf_gene_count', cell_overwrite_ok=True)  # 创建sheet

    # 将数据写入第 i 行，第 j 列
    i = 0
    for data in datas:
        sheet1.write(i, 0, data[0])
        sheet1.write(i, 1, data[1])
        sheet1.write(i, 2, data[2])
        i = i + 1

    f.save(file_path)  # 保存文件


def load_expression_data(dataset, threshold, bins):
    print('	Generating image pairs...')

    wb = xlrd.open_workbook(tf_path)
    tf_dataset = wb.sheet_by_index(0)
    tf_expression_name = []
    for i in range(tf_len):
        tf_expression_name.append(tf_dataset.row_values(i, start_colx=0, end_colx=None))

    wb = xlrd.open_workbook(gene_path)
    gene_dataset = wb.sheet_by_index(0)
    gene_expression_name = []
    for i in range(gene_len):
        gene_expression_name.append(gene_dataset.row_values(i, start_colx=0, end_colx=None))

    tf_threshold = 0 + threshold * 2
    gene_threshold = 0 + threshold * 2

    image_pairs = []

    tf_gene_count = []
    error = []
    expression_data, expression_label = [], []
    for row in tqdm(dataset):
        tf = row[0]
        gene = row[1]
        label = row[2]
        count_pairs = []
        flag = 0
        if gene == 'Dp':
            gene = 'Dp2'
        if tf == 'Dp':
            tf = 'Dp2'
        for i in ['lateral', 'dorsal', 'ventral']:
            tf_path_database_path = database_path + 'tf' + '/' + str(tf) + '/' + i
            gene_path_database_path = database_path + 'gene' + '/' + str(gene) + '/' + i
            tf_files = os.listdir(tf_path_database_path)  # 读入文件夹
            gene_files = os.listdir(gene_path_database_path)  # 读入文件夹
            for p in tf_files:
                for q in gene_files:
                    try:
                        tf_image = read_image(tf_path_database_path + '/' + p)
                        gene_image = read_image(gene_path_database_path + '/' + q)
                        if np.std(tf_image) >= tf_threshold and np.std(gene_image) >= gene_threshold \
                                and tf_image.shape[0] == 128 and gene_image.shape[0] == 128 and np.max(tf_image) > 0 \
                                and np.max(gene_image) > 0:
                            image_pair = [tf_path_database_path + '/' + p, gene_path_database_path + '/' + q, label]
                            image_pairs.append(image_pair)
                            count_pairs.append(image_pair)
                            flag = 1
                    except:
                        continue
        if 0 < len(count_pairs) < 6:
            del image_pairs[-len(count_pairs)::]
            flag = 0
        if flag == 1 and len(count_pairs) > 0:
            flag_tf, flag_gene = 0, 0
            for j in tf_expression_name:
                if j[0] == tf:
                    tf_expression = np.array(j[2:])
                    tf_expression = np.array(list(map(float, tf_expression)))
                    flag_tf = 1
            for j in gene_expression_name:
                if j[0] == gene:
                    gene_expression = np.array(j[2:])
                    gene_expression = np.array(list(map(float, gene_expression)))
                    flag_gene = 1
            if flag_tf == 0 and flag_gene == 0:
                error.append([tf, gene])
            elif flag_tf == 0 and flag_gene == 1:
                error.append([tf, 0])
                # tf_expression = Adf1
            elif flag_tf == 1 and flag_gene == 0:
                error.append([0, gene])
                # gene_expression = olf413
            else:
                H_T_bulk = np.histogram2d(tf_expression, gene_expression, bins=bins)
                H_bulk = H_T_bulk[0].T
                expression_data.append(H_bulk)
                expression_label.append([row[2]])

    expression_data = np.array(expression_data)
    final_expression_data = np.ones((expression_data.shape[0], expression_data.shape[1], expression_data.shape[2], 1))
    final_expression_data[:, :, :, 0] = expression_data
    final_expression_label = np.array(expression_label)

    assert (final_expression_data.shape[0] == final_expression_label.shape[0])
    print("final_expression_data shape:", final_expression_data.shape)
    print("final_expression_label shape:", final_expression_label.shape)
    print("error:", error)

    # 计算正负样本比例
    count = 0
    for i in final_expression_label:
        count = count + i[0]
    print('positive:', count / final_expression_label.shape[0])
    print('negative:', (final_expression_label.shape[0] - count) / final_expression_label.shape[0])

    final_expression_data = normlized(final_expression_data)


    return final_expression_data, final_expression_label


def load_expression_data_for_test_set(dataset, threshold=0, bins=32):
    print('	Generating image pairs...')

    wb = xlrd.open_workbook(tf_path)
    tf_dataset = wb.sheet_by_index(0)
    tf_expression_name = []
    for i in range(tf_len):
        tf_expression_name.append(tf_dataset.row_values(i, start_colx=0, end_colx=None))

    wb = xlrd.open_workbook(gene_path)
    gene_dataset = wb.sheet_by_index(0)
    gene_expression_name = []
    for i in range(gene_len):
        gene_expression_name.append(gene_dataset.row_values(i, start_colx=0, end_colx=None))

    tf_threshold = 0 + threshold * 2
    gene_threshold = 0 + threshold * 2

    image_pairs = []

    tf_gene_count = []
    error = []
    expression_data, expression_label = [], []
    for row in tqdm(dataset):
        tf = row[0]
        gene = row[1]
        label = [row[0], row[1], row[2]]
        count_pairs = []
        flag = 0
        if gene == 'Dp':
            gene = 'Dp2'
        if tf == 'Dp':
            tf = 'Dp2'
        for i in ['lateral', 'dorsal', 'ventral']:
            tf_path_database_path = database_path + 'tf' + '/' + str(tf) + '/' + i
            gene_path_database_path = database_path + 'gene' + '/' + str(gene) + '/' + i
            tf_files = os.listdir(tf_path_database_path)  # 读入文件夹
            gene_files = os.listdir(gene_path_database_path)  # 读入文件夹
            for p in tf_files:
                for q in gene_files:
                    try:
                        tf_image = read_image(tf_path_database_path + '/' + p)
                        gene_image = read_image(gene_path_database_path + '/' + q)
                        if np.std(tf_image) >= tf_threshold and np.std(gene_image) >= gene_threshold \
                                and tf_image.shape[0] == 128 and gene_image.shape[0] == 128 and np.max(tf_image) > 0 \
                                and np.max(gene_image) > 0:
                            image_pair = [tf_path_database_path + '/' + p, gene_path_database_path + '/' + q, label]
                            image_pairs.append(image_pair)
                            count_pairs.append(image_pair)
                            flag = 1
                    except:
                        continue
        if 0 < len(count_pairs) < 6:
            del image_pairs[-len(count_pairs)::]
            flag = 0
        if flag == 1 and len(count_pairs) > 0:
            flag_tf, flag_gene = 0, 0
            for j in tf_expression_name:
                if j[0] == tf:
                    tf_expression = np.array(j[2:])
                    tf_expression = np.array(list(map(float, tf_expression)))
                    flag_tf = 1
            for j in gene_expression_name:
                if j[0] == gene:
                    gene_expression = np.array(j[2:])
                    gene_expression = np.array(list(map(float, gene_expression)))
                    flag_gene = 1
            if flag_tf == 0 and flag_gene == 0:
                error.append([tf, gene])
            elif flag_tf == 0 and flag_gene == 1:
                error.append([tf, 0])
                # tf_expression = Adf1

            elif flag_tf == 1 and flag_gene == 0:
                error.append([0, gene])
                # gene_expression = olf413
            else:
                H_T_bulk = np.histogram2d(tf_expression, gene_expression, bins=bins)
                H_bulk = H_T_bulk[0].T
                expression_data.append(H_bulk)
                expression_label.append(label)

    expression_data = np.array(expression_data)
    final_expression_data = np.ones((expression_data.shape[0], expression_data.shape[1], expression_data.shape[2], 1))
    final_expression_data[:, :, :, 0] = expression_data
    final_expression_label = np.array(expression_label)

    assert (final_expression_data.shape[0] == final_expression_label.shape[0])
    print("final_expression_data shape:", final_expression_data.shape)
    print("final_expression_label shape:", final_expression_label.shape)
    print("error:", error)

    # 计算正负样本比例
    count = 0
    for i in final_expression_label:
        count = count + float(i[2])
    print('positive:', count / final_expression_label.shape[0])
    print('negative:', (final_expression_label.shape[0] - count) / final_expression_label.shape[0])
    final_expression_data = normlized(final_expression_data)

    return final_expression_data, final_expression_label


# 数据标准化处理
def normlized(original_data):
    imgs_normalized = np.empty(original_data.shape)
    imgs_std = np.std(original_data)
    imgs_mean = np.mean(original_data)
    imgs_normalized = (original_data - imgs_mean) / imgs_std
    for i in range(original_data.shape[0]):
        cha = np.max(imgs_normalized[i]) - np.min(imgs_normalized[i])
        if cha == 0:
            cha = np.max(imgs_normalized[i])
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (cha))
    print("imgs_normalized_min: ", np.min(imgs_normalized), "imgs_normalized_max: ", np.max(imgs_normalized))
    assert (np.min(imgs_normalized) >= 0 and np.max(imgs_normalized) == 1)
    # print("imgs_normalized_min: ",np.min(imgs_normalized),"imgs_normalized_max: ",np.max(imgs_normalized))
    return imgs_normalized

# def load_data(train_set, k=0):
#     train_data, train_label = load_image(train_set, k)
#     train_data = normlized(train_data)
#
#     # train_data = train_data.transpose(0, 3, 1, 2)
#     # train_data = train_data * 255.
#     # train_data = clahe_equalized(train_data)
#     # train_data = adjust_gamma(train_data, 1.2)
#     # train_data = train_data / 255.
#     # train_data = train_data.transpose(0, 2, 3, 1)
#     # assert(train_data.shape[3] == 1 and train_data.shape[1] == 256 and  train_data.shape[2] == 320)
#
#     # valid_data, valid_label = load_image(valid_set)
#     # print('valid-set has been loaded!')
#     # test_data, test_label = load_image_for_test_set(test_set)
#     # print('test-set has been loaded!')
#     # '''GTS_data, GTS_label = load_image_for_test_set(GTS_set)
#     # print('GTS_set has been loaded!')'''
#     #
#     # '''train_data /= 255.0
#     # valid_data /= 255.0
#     # test_data /= 255.0
#     # GTS_data /= 255.0'''
#
#     return train_data, train_label

# data, label = load_data('benchmark_dataset_test.xlsx')
# exit()
