# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import random
import copy

input_vec_size = lstm_size = 25  # 输入向量的维度
input_time_mins = 10
pre_time_mins =3
time_step_size = input_time_mins * 20  # 循环层长度
batch_size = 128
test_size = 256


class L1_struct(object):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def batch(self, size):
        data_len = len(self.data)
        index = np.array(range(data_len))
        random.shuffle(index)
        data_batch_loc = index[: size]
        batch_data = self.data[data_batch_loc]
        batch_label = self.label[data_batch_loc]
        batch_fix_data = []
        batch_fix_label = []
        # print(len(batch_data),batch_data.shape)
        for i in range(len(batch_data)):
            temp_len = len(batch_data[i])
            temp_index = np.random.randint(low = 0, high = temp_len-time_step_size)
            batch_fix_data.append(batch_data[i][temp_index : temp_index+time_step_size])
            batch_fix_label.append(batch_label[i][temp_index+time_step_size-1])
#             print(batch_data[i].shape)
        return np.array(batch_fix_data), np.array(batch_fix_label)


def data_to_label(trunc_data):
    label = []
    for i in range(len(trunc_data[: -pre_time_mins*20, 0])):
        temp = trunc_data[i+1: i+pre_time_mins*20+1, 0]
        n = trunc_data[i, 0]
        h = np.max(temp)
        h_index = np.where(temp == h)[0][0]
        l = np.min(temp)
        l_index = np.where(temp == l)[0][0]
        r1 = (h - n)/n
        r2 = (l - n)/n
        if n == 0:
            print ('trunc bug', n, l, h, trunc_data[i, -1])
        if r1 < 0.0025 and r2 > -0.0025:
            label.append([1, 0, 0])
        elif r1 >= 0.0025:
            if r2 > -0.0025:
                label.append([0, 1, 0])
            if r2 <= -0.0025 and h_index < l_index:
                label.append([0, 1, 0])
        else:
            label.append([0, 0, 1])

    norm_len = 5 * 20
    copy_data = copy.deepcopy(trunc_data)
    mean_price = np.mean(copy_data[:norm_len, 0])
    std_price = np.std(copy_data[:norm_len, 0])
    mean_volume = np.mean(copy_data[:norm_len, 1])
    std_volume = np.std(copy_data[:norm_len, 1])
    mean_ba = np.mean(copy_data[:norm_len, 15:25])
    std_ba = np.std(copy_data[:norm_len, 15:25])
    # print (std_price, std_volume)
    copy_data[:, 0] = (copy_data[:, 0] - mean_price) / std_price
    copy_data[:, 1] = (copy_data[:, 1] - mean_volume) / std_volume
    for i in range(5, 15):
        copy_data[:, i] = (copy_data[:, i] - mean_price) / std_price
    for i in range(15, 25):
        copy_data[:, i] = (copy_data[:, i] - mean_ba) / std_ba
    copy_data = copy_data[norm_len:]
    label = np.array(label)[norm_len:]
    copy_data = copy_data[: len(label)]

    return copy_data, label

data = sio.loadmat('SZ002104.mat')
keys = ['Price', 'Volume', 'BSFlag', 'AskPrice5', 'BidPrice5', 'AskVolume5', 'BidVolume5']
time_s = np.array([i[0] for i in data['Time']])
temp = np.array([0 for i in range(len(time_s))])
temp[time_s > 94500000] = 1
day_b = []
for i in range(len(time_s)):
    if temp[i]==0 and temp[i+1] == 1:
        day_b.append(i)
temp = np.array([1 for i in range(len(time_s))])
temp[time_s < 145000000] = 0
day_e = []
for i in range(len(time_s)):
    if temp[i]==0 and temp[i+1]==1:
        day_e.append(i)
list_data = [[data[i][j] for i in keys] for j in range(len(data['Price']))]
list_data = np.array(list_data)
print(list_data.shape)
del data



# 还没有正则化

np_data = []
for temp1 in list_data:
    t = []
    for temp2 in temp1:
        for temp3 in temp2:
            if temp3 == 'B':
                t = t + [1, 0, 0]
                continue
            elif temp3 == 'S':
                t = t + [0, 1, 0]
                continue
            elif temp3 == ' ':
                t = t + [0, 0, 1]
                continue
            else:
                t.append(np.float(temp3))
    np_data.append(t)
np_data = np.array(np_data)
del list_data

all_data = []
all_label = []
# for day in range(len(day_b)):
for day in range(2):
    print(day_b[day], day_e[day])
    day_data, day_label = data_to_label(np_data[day_b[day]:day_e[day]])
    all_data.append(day_data)
    all_label.append(day_label)

all_data = np.array(all_data)
all_label = np.array(all_label)
all_data_size = len(all_data)
tr_size = int(all_data_size * 0.7)
te_size = all_data_size - tr_size
data_index = np.arange(all_data_size)
np.random.shuffle(data_index)
train_indices = data_index[:tr_size]
trX = all_data[train_indices]
trY = all_label[train_indices]
test_indices = data_index[tr_size:]
teX = all_data[test_indices]
teY = all_label[test_indices]

tr_L1_data = L1_struct(trX, trY)
te_L1_data = L1_struct(teX, teY)

view_data, view_label = all_data[0], all_label[0]
plt.plot(view_data[:,0])
buy_index = []
sell_index = []
for i in range(len(view_label)):
    if list(view_label[i]) == [0, 1, 0]:
        buy_index.append(i)
    if list(view_label[i]) == [0, 0, 1]:
        sell_index.append(i)
plt.scatter(buy_index, view_data[buy_index, 0], c='r')
plt.scatter(sell_index, view_data[sell_index, 0], c='y')
plt.show()