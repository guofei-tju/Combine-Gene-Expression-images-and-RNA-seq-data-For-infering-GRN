import os
import csv
import string
import numpy as np
from expression_data import load_expression_data
from expression_data import load_expression_data_for_test_set
from expression_model import expression_model
import h5py

import xlwt,xlrd


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


def write_hdf5(arr,fpath):
  with h5py.File(fpath,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)


dataset_path = 'new_benchmark_dataset.xlsx'

wb = xlrd.open_workbook(dataset_path)
new_benchmark_dataset = wb.sheet_by_index(0)
dataset = []
for i in range(5566):
    row = new_benchmark_dataset.row_values(i, start_colx=0, end_colx=None)
    label = float(row[2])
    new_row = [row[0], row[1], label]
    dataset.append(new_row)

print('dataset length: ', len(dataset))

five_fold = []
for i in range(5):
    if i < 4:
        five_fold.append(dataset[(i * int(len(dataset) / 5)):((i + 1) * int(len(dataset) / 5))])
    else:
        five_fold.append(dataset[(i * int(len(dataset) / 5))::])

outcome = np.zeros((5, 8))
m = 0
for k in [64]:
    for i in range(1):
        print('Round ' + str(i + 1) + ' starts now!')
        test_set = five_fold[i]

        train_all_set = []
        for j in range(5):
            if j == i:
                continue
            train_all_set += five_fold[j][:]

        train_set = train_all_set[0:int(len(train_all_set) * 9 / 10)]
        valid_set = train_all_set[int(len(train_all_set) * 9 / 10):len(train_all_set)]

        print('train_set_len:', len(train_set), 'valid_set_len:', len(valid_set), 'test_set_len:', len(test_set))
        train_expression_data, train_expression_label = load_expression_data(train_set, threshold=0, bins=k)
        print('train-set has beed loaded!')

        valid_expression_data, valid_expression_label = load_expression_data(valid_set, threshold=0, bins=k)
        print('valid_set has beed loaded!')

        test_expression_data, test_expression_label = load_expression_data_for_test_set(test_set, threshold=0, bins=k)
        print('test-set has beed loaded!')
        outcome[i, :] = expression_model(i, train_expression_data, train_expression_label, valid_expression_data,
                                         valid_expression_label, test_expression_data, test_expression_label)
with open('./output/outcome.txt', 'w') as f:
    f.write('test_acc	test_f1  AUC   AUPR  TN  TP  FN  FP\n')
    for row in outcome:
        f.write(str(row[0]) + '\t' + str(row[1]) + '\t' + str(row[2]) + '\t' + str(row[3]) +
                '\t' + str(row[4]) + '\t' + str(row[5]) + '\t' + str(row[6]) + '\t' + str(row[7]) +'\n')
f.close()


