# coding=utf-8
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
bins = 64


def clahe_equalized(imgs):
    assert (len(imgs.shape) == 4)  # 3D arrays
    assert (imgs.shape[1] == 1)  # check the channel is 1
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i, 0] = clahe.apply(np.array(imgs[i, 0], dtype=np.uint8))
    return imgs_equalized


def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape) == 4)  # 4D arrays
    assert (imgs.shape[1] == 1)  # check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i, 0] = cv2.LUT(np.array(imgs[i, 0], dtype=np.uint8), table)
    return new_imgs


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


def load_image(dataset,threshold):
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
    for row in dataset:
        tf = row[0]
        gene = row[1]
        label = row[2]
        count_pairs = []
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
                        if np.std(tf_image) >= tf_threshold and np.std(gene_image) >= gene_threshold:
                            image_pair = [tf_path_database_path + '/' + p, gene_path_database_path + '/' + q, label]
                            image_pairs.append(image_pair)
                            count_pairs.append(image_pair)
                    except:
                        continue
        if 0 < len(count_pairs) < 6:
            del image_pairs[-len(count_pairs)::]
        elif len(count_pairs) >= 6:
            tf_gene_count.append([tf, gene, len(count_pairs)])

    print("connect_image_pairs_len:", len(tf_gene_count))

    random.shuffle(image_pairs)
    print('	Image pair list is ready: ' + str(len(image_pairs)))
    print('	Read images...')

    # 计算正负样本比例
    count = 0
    for i in image_pairs:
        count = count + i[2]
    print('positive:', count / len(image_pairs))
    print('negative:', (len(image_pairs) - count) / len(image_pairs))

    data = np.zeros((len(image_pairs), 256, 320, 1), dtype="float64")  # len(image_pairs)
    label = np.zeros((len(image_pairs), 1), dtype="float64")
    error_count = 0
    k=0
    error = []
    expression_data,expression_label = [], []
    for i, row in enumerate(tqdm(image_pairs)):
        tf_image = read_image(row[0])
        gene_image = read_image(row[1])
        if tf_image.shape[0] == 128 and gene_image.shape[0] == 128 and np.max(tf_image) > 0 and np.max(gene_image) > 0:
                # and np.std(tf_image) >= tf_threshold and np.std(gene_image) >= gene_threshold:
            label[k, 0] = row[2]
            data[k, 0:128, :, 0] = np.copy(tf_image)
            data[k, 128:256, :, 0] = np.copy(gene_image)
            k = k + 1

            tf = row[0].split('/')[-3]
            gene = row[1].split('/')[-3]
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
            elif flag_tf == 1 and flag_gene == 0:
                error.append([0, gene])
            else:
                H_T_bulk = np.histogram2d(tf_expression, gene_expression, bins=bins)
                H_bulk = H_T_bulk[0].T

            expression_data.append(H_bulk)
            expression_label.append([row[2]])
        else:
            error_count = error_count + 1

    data = data[0:len(image_pairs) - error_count, :, :, :]
    label = label[0:len(image_pairs) - error_count, :]

    expression_data = np.array(expression_data)
    final_expression_data = np.ones((expression_data.shape[0], expression_data.shape[1], expression_data.shape[2], 1))
    final_expression_data[:, :, :, 0] = expression_data
    final_expression_label = np.array(expression_label)

    assert (final_expression_data.shape[0] == final_expression_label.shape[0])
    assert (final_expression_data.shape[0] == data.shape[0])
    assert (data.shape[0] == label.shape[0])
    print("final_expression_data shape:", final_expression_data.shape)
    print("final_expression_label shape:", final_expression_label.shape)
    print("error:", error)
    print('connect_image_data = ', data.shape)
    print('connect_image_label = ', label.shape)

    return data, label, final_expression_data, final_expression_label


def load_image_for_test_set(dataset, threshold=0):
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

    tf_gene_count = []
    image_pairs = []
    # database_path = './new_training/second_mannual_del_image_storage/'
    for row in dataset:
        tf = row[0]
        gene = row[1]
        label = [row[0], row[1], row[2]]
        count_pairs=[]
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
                        if np.std(tf_image) >= tf_threshold and np.std(gene_image) >= gene_threshold:
                            image_pair = [tf_path_database_path + '/' + p, gene_path_database_path + '/' + q, label]
                            image_pairs.append(image_pair)
                            count_pairs.append(image_pair)
                    except:
                        continue

        if 0 < len(count_pairs) < 6:
            del image_pairs[-len(count_pairs)::]
        elif len(count_pairs) >= 6:
            tf_gene_count.append([tf, gene, len(count_pairs)])

    print("connect_image_pairs_len:", len(tf_gene_count))

    random.shuffle(image_pairs)
    print('	Image pair list is ready: ' + str(len(image_pairs)))
    print('	Read images...')

    # 计算正负样本比例
    count = 0
    for i in image_pairs:
        count = count + i[2][2]
    print('positive:', count / len(image_pairs))
    print('negative:', (len(image_pairs) - count) / len(image_pairs))

    data = np.zeros((len(image_pairs), 256, 320, 1), dtype="float64")  # len(image_pairs)
    label = []
    error_count = 0
    k=0
    error = []
    expression_data, expression_label = [], []
    for i, row in enumerate(tqdm(image_pairs)):
        # print(str(row[0]), str(row[1]))
        tf_image = read_image(row[0])
        gene_image = read_image(row[1])
        if tf_image.shape[0] == 128 and gene_image.shape[0] == 128 and np.max(tf_image) > 0 and np.max(gene_image) > 0 \
                and np.std(tf_image) >= tf_threshold and np.std(gene_image) >= gene_threshold:
            label.append(row[2])
            data[k, 0:128, :, 0] = np.copy(tf_image)
            data[k, 128:256, :, 0] = np.copy(gene_image)
            k = k + 1

            tf = row[0].split('/')[-3]
            gene = row[1].split('/')[-3]
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
            elif flag_tf == 1 and flag_gene == 0:
                error.append([0, gene])
            else:
                H_T_bulk = np.histogram2d(tf_expression, gene_expression, bins=bins)
                H_bulk = H_T_bulk[0].T

            expression_data.append(H_bulk)
            expression_label.append(row[2])
        else:
            error_count = error_count + 1
    data = data[0:len(image_pairs) - error_count, :, :, :]

    expression_data = np.array(expression_data)
    final_expression_data = np.ones((expression_data.shape[0], expression_data.shape[1], expression_data.shape[2], 1))
    final_expression_data[:, :, :, 0] = expression_data
    final_expression_label = np.array(expression_label)
    assert (final_expression_data.shape[0] == final_expression_label.shape[0])
    assert (final_expression_data.shape[0] == data.shape[0])
    assert (data.shape[0] == len(label))
    print("final_expression_data shape:", final_expression_data.shape)
    print("final_expression_label shape:", final_expression_label.shape)
    print("error:", error)
    print('connect_image_data = ', data.shape)
    print('connect_image_label = ', len(label))
    print("tf_threshold:", tf_threshold, "gene_threshold", gene_threshold)

    data = normlized(data)
    final_expression_data = normlized(final_expression_data)

    return data, label, final_expression_data, final_expression_label


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


def load_data(train_set, k=0):
    train_data, train_label, final_expression_data, final_expression_label = load_image(train_set, k)
    train_data = normlized(train_data)
    final_expression_data = normlized(final_expression_data)

    # train_data = train_data.transpose(0, 3, 1, 2)
    # train_data = train_data * 255.
    # train_data = clahe_equalized(train_data)
    # train_data = adjust_gamma(train_data, 1.2)
    # train_data = train_data / 255.
    # train_data = train_data.transpose(0, 2, 3, 1)
    # assert(train_data.shape[3] == 1 and train_data.shape[1] == 256 and  train_data.shape[2] == 320)

    # valid_data, valid_label = load_image(valid_set)
    # print('valid-set has been loaded!')
    # test_data, test_label = load_image_for_test_set(test_set)
    # print('test-set has been loaded!')
    # '''GTS_data, GTS_label = load_image_for_test_set(GTS_set)
    # print('GTS_set has been loaded!')'''
    #
    # '''train_data /= 255.0
    # valid_data /= 255.0
    # test_data /= 255.0
    # GTS_data /= 255.0'''

    return train_data, train_label, final_expression_data, final_expression_label

# data, label = load_data('benchmark_dataset_test.xlsx')
# exit()
