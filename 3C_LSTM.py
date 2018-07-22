# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import random
import copy

input_vec_size = lstm_size = 23  # 输入向量的维度
input_time_mins = 30
pre_time_mins = 10
time_step_size = input_time_mins * 20  # 循环层长度
batch_size = 20


def view_s(data,label):
    # 可视化检查
    for i in range(len(data)):
        fig = plt.figure()
        ax1 = fig.add_subplot(321)
        ax1.plot(data[i][:, 0])
        buy_index = []
        sell_index = []
        for j in range(len(label[i])):
            if label[i][j] == 1:
                buy_index.append(j)
            if label[i][j] == 0:
                sell_index.append(j)
        ax1.scatter(buy_index, data[i][buy_index, 0], c='r')
        ax1.scatter(sell_index, data[i][sell_index, 0], c='y')

        ax4 = fig.add_subplot(322)
        ax4.plot(data[i][:, 1])
        ax5 = fig.add_subplot(323)
        ax5.plot(data[i][:, 3:8])
        ax6 = fig.add_subplot(324)
        ax6.plot(data[i][:, 8:13])
        ax7 = fig.add_subplot(325)
        ax7.plot(data[i][:, 13:18])
        ax8 = fig.add_subplot(326)
        ax8.plot(data[i][:, 18:23])
        plt.show()

def norm_everyday(data):
    for i in range(len(data)):
        norm_len = len(data[i][:,0])
        mean_price = np.mean(data[i][:norm_len, 0])
        std_price = np.std(data[i][:norm_len, 0])
        mean_volume = np.mean(data[i][:norm_len, 1])
        std_volume = np.std(data[i][:norm_len, 1])
        mean_ba = np.mean(data[i][:norm_len, 13:])
        std_ba = np.std(data[i][:norm_len, 13:])
        data[i][:, 0] = (data[i][:, 0] - mean_price) / std_price
        data[i][:, 1] = (data[i][:, 1] - mean_volume) / std_volume
        data[i][:, 3:13] = (data[i][:, 3:13] - mean_price) / std_price
        data[i][:, 13:23] = (data[i][:, 13:23] - mean_ba) / std_ba
    return data


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

    def batch(self, size, check = False):
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
            if p > 0.51:
                # print(batch_label[batch])
                p_index = [idx for (idx, val) in enumerate(batch_label[batch]) if val == 0 and idx>200]
                if p_index != []:
                    temp_index = p_index[np.random.randint(low=0, high=len(p_index))]+1

            copy_data = copy.deepcopy(batch_data[batch][: temp_index, :])
            sequence_length.append(len(copy_data))
            batch_fix_data.append(copy_data)
            if batch_label[batch][temp_index-1] == 0:
                batch_fix_label.append([0, 1])
            else:
                batch_fix_label.append([1, 0])
            # print("BATCH %d DONE", (batch))

        batch_fix_data = np.array(batch_fix_data)
        batch_fix_data = apply_to_zeros(batch_fix_data)
        return batch_fix_data, np.array(batch_fix_label), np.array(sequence_length)


# 载入数据进行部分预处理
data = sio.loadmat('SZ300112.mat')
use_keys = ['Time', 'Price', 'Volume', 'BSFlag', 'AskPrice5', 'BidPrice5', 'AskVolume5', 'BidVolume5']
data_dic = {}
for keys in list(data.keys())[3:]:
    if keys in use_keys:
        data_dic[keys] = list(data[keys])
        # print(len(list(data[keys])))

data = pd.DataFrame(data_dic)
w = 20
bsflg2num = {'B': 0, 'S': 1, ' ': 2}
data['BSFlag'] = data['BSFlag'].apply(lambda x: bsflg2num[x])
data['Volume'] = data['Volume'].apply(lambda x: np.float(x[0]))
data['Price'] = data['Price'].apply(lambda x: np.float(x[0]))
data['Time'] = data['Time'].apply(lambda x: x[0])
data['Time_flag'] = data['Time'].apply(lambda x: 1 if x>94500000 and x<144500000 else 0)
data['Time_diff'] = data['Time_flag'].diff()
close_price = list(data['Price'][data['Time_diff'] == -1])
day_b = np.array(data[data['Time_diff'] == 1].index)
day_e = np.array(data[data['Time_diff'] == -1].index)
data['label1'] = data.Price.rolling(window=20*w).mean().shift(-20*w)/data['Price'] - 1
data['label2'] = data.Price.rolling(window=20*w).max().shift(-20*w)/data['Price'] - 1
data['label3'] = data.Price.rolling(window=20*w).min().shift(-20*w)/data['Price'] - 1
data['label'] = data.apply(lambda x: 0 if abs(x['label1']) < 0.004 and x['label2'] < 0.006 and x['label3'] > -0.006
                                      else 1, axis=1)

for i in range(0, 5):
    data['AskPrice' + '_t' + str(i)] = data['AskPrice5'].apply(lambda x: np.float(x[i]))
    data['BidPrice' + '_t' + str(i)] = data['BidPrice5'].apply(lambda x: np.float(x[i]))
    data['AskVolume' + '_t' + str(i)] = data['AskVolume5'].apply(lambda x: np.float(x[i]))
    data['BidVolume' + '_t' + str(i)] = data['BidVolume5'].apply(lambda x: np.float(x[i]))
del data['AskPrice5'], data['BidPrice5'], data['AskVolume5'], data['BidVolume5']

f2use = ['Price', 'Volume', 'BSFlag',
         'AskPrice_t0', 'AskPrice_t1', 'AskPrice_t2', 'AskPrice_t3', 'AskPrice_t4',
         'BidPrice_t0', 'BidPrice_t1', 'BidPrice_t2', 'BidPrice_t3', 'BidPrice_t4',
         'AskVolume_t0', 'AskVolume_t1', 'AskVolume_t2', 'AskVolume_t3', 'AskVolume_t4',
         'BidVolume_t0', 'BidVolume_t1', 'BidVolume_t2', 'BidVolume_t3', 'BidVolume_t4',
         ]
all_data = []
all_label = []
for day in range(1,51):
    print('day:', day)
    day_data = np.array(data[f2use][day_b[day]:day_e[day]])
    day_data[:, 0] = day_data[:, 0]/close_price[day-1]
    day_data[:, 3:13] = day_data[:, 3:13]/close_price[day-1]
    # print(day_data.shape)
    day_label = np.array(data['label'][day_b[day]:day_e[day]])
    all_data.append(day_data)
    all_label.append(day_label)

all_data = np.array(all_data)
all_label = np.array(all_label)
trX, teX, trY, teY = split_tr_te(all_data, all_label)
# view_s(trX,trY)
trX = norm_everyday(trX)
teX = norm_everyday(teX)
# view_s(pp, trY)
tr_L1_data = L1_struct(trX, trY)
te_L1_data = L1_struct(teX, teY)
# tr_L1_data.batch(50)

X = tf.placeholder("float", [None, None, lstm_size])
Y = tf.placeholder("float", [None, 2])
sequence_length = tf.placeholder(tf.int32, [None])

def model(X):
    lstm = rnn.BasicLSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=True)
    # mlstm_cell = rnn.MultiRNNCell([lstm]*4)
    outputs, last_states = tf.nn.dynamic_rnn(
        cell=lstm,
        dtype=tf.float32,
        sequence_length=sequence_length,
        inputs=X)
    # return last_states.h, last_states.c  # State size to initialize the stat
    return last_states.c, last_states.c  # State size to initialize the stat

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))
# get lstm_size and output 10 labels
w_1 = init_weights([lstm_size, 32])
b_1 = init_weights([32])
w_2 = init_weights([32, 32])
b_2 = init_weights([32])
w_3 = init_weights([32, 32])
b_3 = init_weights([32])
w_out = init_weights([32, 2])
b_out = init_weights([2])

cnn_output, states = model(X)
h1 = tf.nn.relu(tf.matmul(cnn_output, w_1) + b_1)
h2 = tf.nn.relu(tf.matmul(h1, w_2) + b_2)
h3 = tf.nn.relu(tf.matmul(h2, w_3) + b_3)
py_x = tf.nn.softmax(tf.matmul(h2, w_out) + b_out)


cost = -tf.reduce_sum(Y*tf.log(tf.clip_by_value(py_x,1e-5,1)))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
correct_prediction = tf.equal(tf.argmax(py_x, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

session_conf = tf.ConfigProto()
session_conf.gpu_options.allow_growth = True

# Launch the graph in a session
with tf.Session(config=session_conf) as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(50):
        for t in range(20):
            batch_data, batch_label, seq = tr_L1_data.batch(50)
            sess.run(train_op, feed_dict={X: batch_data, Y: batch_label, sequence_length: seq})
            if t % 10 == 0:
                print('train_acc', sess.run(accuracy, feed_dict={X: batch_data, Y: batch_label, sequence_length: seq}))

        batch_data, batch_label, seq= te_L1_data.batch(20)
        print('last_states', sess.run(states, feed_dict={X: batch_data, Y: batch_label, sequence_length: seq}))
        print(i, sess.run(accuracy, feed_dict={X: batch_data, Y: batch_label, sequence_length: seq}))
        p = sess.run(py_x, feed_dict={X: batch_data, Y: batch_label, sequence_length: seq})
        # print(batch_data)
        p = np.concatenate((p, batch_label), 1)
        print(p)