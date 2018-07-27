# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import random
import copy



def view_s(data, label):
    fig = plt.figure()
    ax1 = fig.add_subplot(331)
    ax1.plot(data[:, 0])
    buy_index = []
    sell_index = []
    for j in range(len(label)):
        if label[j] == 1:
            buy_index.append(j)
        if label[j] == 0:
            sell_index.append(j)
    ax1.scatter(buy_index, data[buy_index, 0], c='r')
    ax1.scatter(sell_index, data[sell_index, 0], c='y')
    ax2 = fig.add_subplot(332)
    ax2.plot(data[:, 1])
    ax3 = fig.add_subplot(333)
    ax3.plot(data[:, 3:8])
    ax4 = fig.add_subplot(334)
    ax4.plot(data[:, 8:13])
    ax5 = fig.add_subplot(335)
    ax5.plot(data[:, 13:18])
    ax6 = fig.add_subplot(336)
    ax6.plot(data[:, 18:23])
    ax7 = fig.add_subplot(337)
    ax7.plot(data[:, 25])
    plt.show()
    return


def norm_data(data):
    mean_price = np.mean(data[:, 0])
    std_price = np.std(data[:, 0])
    mean_volume = np.mean(data[:, 1])
    std_volume = np.std(data[:, 1])
    mean_ba = np.mean(data[:, 13:23])
    std_ba = np.std(data[:, 13:23])
    mean_price_std = np.mean(data[:, 25])
    std_price_std = np.std(data[:, 25])
    data[:, 0] = (data[:, 0] - mean_price) / std_price
    data[:, 1] = (data[:, 1] - mean_volume) / std_volume
    data[:, 3:13] = (data[:, 3:13] - mean_price) / std_price
    data[:, 13:23] = (data[:, 13:23] - mean_ba) / std_ba
    data[:, 25] = (data[:, 25] - mean_price_std) / std_price_std
    return data, [mean_price, mean_ba, mean_volume, mean_price]


def norm_data11(data):
    min_price = np.min(data[:, 0])
    max_price = np.max(data[:, 0])
    min_volume = np.min(data[:, 1])
    max_volume = np.max(data[:, 1])
    min_ba = np.min(data[:, 13:23])
    max_ba = np.max(data[:, 13:23])
    min_price_std = np.min(data[:, 25])
    max_price_std = np.max(data[:, 25])
    data[:, 0] = (data[:, 0] - min_price) / (max_price - min_price) - 0.5
    data[:, 1] = (data[:, 1] - min_volume) / (max_volume - min_volume) - 0.5
    data[:, 3:13] = (data[:, 3:13] - min_price) / (max_price - min_price) - 0.5
    data[:, 13:23] = (data[:, 13:23] - min_ba) / (max_ba -min_ba) - 0.5
    data[:, 25] = (data[:, 25] - min_price_std) / (max_price_std -min_price_std) - 0.5
    return data, [min_price, min_volume, min_ba, min_price_std]


def split_tr_te(data, label, size=0.8):
    tr_size = int(len(data) * size)
    data_index = np.arange(len(data))
    np.random.shuffle(data_index)
    tr_indices = data_index[:tr_size]
    te_indices = data_index[tr_size:]
    trX = data[tr_indices]
    teX = data[te_indices]
    trY = label[tr_indices]
    teY = label[te_indices]

    return trX, teX, trY, teY


def apply_to_zeros(lst):
    inner_max_len = max(map(len, lst))
    # print (lst.shape, lst[0].shape, lst[0][0].shape,)
    result = np.zeros([len(lst), inner_max_len, len(lst[0][0])])
    for i, row in enumerate(lst):
        for j, val in enumerate(row):
            for k, temp in enumerate(val):
                result[i][j][k] = temp
    return result


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
        sequence_length = []
        for batch in range(len(batch_data)):
            p = np.random.random()
            temp_index = np.random.randint(low=200, high=len(batch_data[batch]))
            if p > 0.5:
                p_index = [idx for (idx, val) in enumerate(batch_label[batch]) if val == 1 and idx>200]
                if p_index != []:
                    temp_index = p_index[np.random.randint(low=0, high=len(p_index))]

            copy_data = copy.deepcopy(batch_data[batch][: temp_index+1, :])
            copy_label = copy.deepcopy(batch_label[batch][: temp_index+1])
            copy_data, temp = norm_data11(copy_data)
            # view_s(copy_data, copy_label)
            sequence_length.append(len(copy_data))
            batch_fix_data.append(copy_data)
            if copy_label[-1] == 0:
                batch_fix_label.append([0, 1])
            else:
                batch_fix_label.append([1, 0])
            # print("BATCH %d DONE", (batch))

        batch_fix_data = np.array(batch_fix_data)
        batch_fix_data = apply_to_zeros(batch_fix_data)
        return batch_fix_data, np.array(batch_fix_label), np.array(sequence_length)