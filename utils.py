# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import random
import copy


def Cdata2label(trunc_data, pre_time_mins):
    label = []
    for i in range(len(trunc_data[: -pre_time_mins*20, 0])):
        temp = trunc_data[i+1: i+pre_time_mins*20+1, 0]
        n = trunc_data[i, 0]
        m = np.mean(temp)
        h = np.max(temp)
        l = np.min(temp)
        r1 = (h - n)/n
        r2 = (l - n)/n
        r3 = (m - n)/n
        if n == 0:
            print ('trunc bug', n, l, h, trunc_data[i, -1])

        if r3 > 0.004 and r2 > -0.002:
            label.append([0, 1, 0])
        elif r3 < -0.004 and r1 < 0.002:
            label.append([0, 0, 1])
        else:
            label.append([1, 0, 0])

    copy_data = copy.deepcopy(trunc_data)
    tag = 0
    filter_label = []
    while (tag <= len(label)-3):
        if list(label[tag]) != [1, 0, 0]:
            if list(label[tag]) == list(label[tag+1]) and list(label[tag]) == list(label[tag+2]):
                filter_label.append(label[tag])
                filter_label.append(label[tag])
                filter_label.append(label[tag])
                tag = tag+3
                continue
            else:
                filter_label.append([1, 0, 0])
                filter_label.append([1, 0, 0])
                filter_label.append([1, 0, 0])
                tag = tag + 3
                continue
        else:
            filter_label.append([1, 0, 0])
            tag = tag + 1
    label = np.array(label)
    filter_label.append([1, 0, 0])
    filter_label.append([1, 0, 0])
    filter_label = np.array(filter_label)
    copy_data = copy_data[: len(label)]
    return copy_data, label, filter_label


def MMdata2label(price, pre_time_mins):
    label = []
    for i in range(len(price[: -pre_time_mins*20, 0])):
        temp = price[i+1: i+pre_time_mins*20+1, 0]
        n = price[i, 0]
        m = np.mean(temp)
        h = np.max(temp)
        l = np.min(temp)
        r1 = (h - n)/n
        r2 = (l - n)/n
        r3 = (m - n)/n
        if n == 0:
            print ('trunc bug', n, l, h, price[i, -1])

        if r3 > 0.004 and r2 > -0.002:
            label.append([0, 1])
        elif r3 < -0.004 and r1 < 0.002:
            label.append([0, 1])
        else:
            label.append([1, 0])

    trunc_data = price[: len(label)]
    day_data = copy.deepcopy(trunc_data)
    label = np.array(label)
    return day_data, label