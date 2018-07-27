# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import scipy.io as sio
from utils import *
import pandas as pd

input_vec_size = 10
hiede_size = 256  # 输入向量的维度



# 载入数据进行部分预处理
data = sio.loadmat('SZ300112.mat')
use_keys = ['Time', 'Price', 'Volume', 'BSFlag', 'AskPrice5', 'BidPrice5', 'AskVolume5', 'BidVolume5']
data_dic = {}
for keys in list(data.keys())[:]:
    if keys in use_keys:
        data_dic[keys] = list(data[keys])
        # print(len(list(data[keys])))

data = pd.DataFrame(data_dic)
w = 2
bsflg2num = {'B': 0, 'S': 1, ' ': 2}
data['Price'] = data['Price'].apply(lambda x: np.float(x[0]))
data['lPrice'] = data['Price'].shift()
data['tick_r'] = (data['Price'] - data['lPrice'])/data['Price']
data['AskPrice5'] = data['AskPrice5'].apply(lambda x: np.array(x))
data['BidPrice5'] = data['BidPrice5'].apply(lambda x: np.array(x))
data['AskVolume5'] = data['AskVolume5'].apply(lambda x: np.array(x))
data['BidVolume5'] = data['BidVolume5'].apply(lambda x: np.array(x))
data['sum_apv'] = data['AskPrice5']*data['AskVolume5']
data['sum_apv'] = data['sum_apv'].apply(lambda x: np.sum(x))
data['sum_bpv'] = data['BidPrice5']*data['BidVolume5']
data['sum_bpv'] = data['sum_bpv'].apply(lambda x: np.sum(x))
data['sum_av'] = data['AskVolume5'].apply(lambda x: np.sum(x))
data['sum_bv'] = data['BidVolume5'].apply(lambda x: np.sum(x))
data['v_ratio'] = data['sum_av']/data['sum_bv']
data['pv_ratio'] = data['sum_apv']/data['sum_bpv']
#
# print(data[['tick_r', 'pv_ratio', 'v_ratio']])
# print(data[['AskPrice5', 'AskVolume5']])
# print(data[['BidPrice5', 'BidVolume5']])
data['BSFlag'] = data['BSFlag'].apply(lambda x: bsflg2num[x])
data['BSFlag1'] = data['BSFlag'].apply(lambda x: 1 if x == 0 else 0)
data['BSFlag2'] = data['BSFlag'].apply(lambda x: 1 if x == 1 else 0)
data['BSFlag3'] = data['BSFlag'].apply(lambda x: 1 if x == 2 else 0)
data['Volume'] = data['Volume'].apply(lambda x: 0 if x == 0 else np.log(np.float(x[0])))
data['Time'] = data['Time'].apply(lambda x: x[0])
data['Time_flag'] = data['Time'].apply(lambda x: 1 if x>94500000 and x<144500000 else 0)
data['Time_diff'] = data['Time_flag'].diff()
close_price = list(data['Price'][data['Time_diff'] == -1])
day_b = np.array(data[data['Time_diff'] == 1].index)
day_e = np.array(data[data['Time_diff'] == -1].index)
data['mp'] = data.Price
data['Price'] = data.Price.rolling(window=20).mean()
data['std_Price'] = data['Price'].rolling(window=40).mean()
data['label1'] = data.mp.rolling(window=20).mean().shift(-20*w)/data['Price'] - 1
data['label2'] = data.mp.rolling(window=20).max().shift(-20*w)/data['Price'] - 1
data['label3'] = data.mp.rolling(window=20).min().shift(-20*w)/data['Price'] - 1
data['label'] = data.apply(lambda x: 0 if abs(x['label1']) < 0.002 and x['label2'] < 0.003 and x['label3'] > -0.003
                                      else 1, axis=1)
# data['label'] = data.apply(lambda x: 0 if abs(x['label1']) < 0.002 else 1, axis=1)

for i in range(0, 5):
    data['AskPrice' + '_t' + str(i)] = data['AskPrice5'].apply(lambda x: np.float(x[i]))
    data['BidPrice' + '_t' + str(i)] = data['BidPrice5'].apply(lambda x: np.float(x[i]))
    data['AskVolume' + '_t' + str(i)] = data['AskVolume5'].apply(lambda x: 0 if x[i] == 0 else np.log(np.float(x[i])))
    data['BidVolume' + '_t' + str(i)] = data['BidVolume5'].apply(lambda x: 0 if x[i] == 0 else np.log(np.float(x[i])))
del data['AskPrice5'], data['BidPrice5'], data['AskVolume5'], data['BidVolume5']

f2use = ['Price', 'Volume', 'BSFlag1',
         # 'AskPrice_t0', 'AskPrice_t1', 'AskPrice_t2', 'AskPrice_t3', 'AskPrice_t4',
         # 'BidPrice_t0', 'BidPrice_t1', 'BidPrice_t2', 'BidPrice_t3', 'BidPrice_t4',
         # 'AskVolume_t0', 'AskVolume_t1', 'AskVolume_t2', 'AskVolume_t3', 'AskVolume_t4',
         # 'BidVolume_t0', 'BidVolume_t1', 'BidVolume_t2', 'BidVolume_t3', 'BidVolume_t4',
         'BSFlag2', 'BSFlag3', 'std_Price', 'mp', 'tick_r', 'pv_ratio', 'v_ratio'
         ]
all_data = []
all_label = []
for day in range(1, 201):
    print('day:', day)
    day_data = np.array(data[f2use][day_b[day]:day_e[day]])
    # day_data[:, 0] = day_data[:, 0]/close_price[day-1]
    # day_data[:, 3:13] = day_data[:, 3:13]/close_price[day-1]
    # print(day_data.shape)
    day_label = np.array(data['label'][day_b[day]:day_e[day]])
    all_data.append(day_data)
    all_label.append(day_label)

all_data = np.array(all_data)
all_label = np.array(all_label)
trX, teX, trY, teY = split_tr_te(all_data, all_label)
tr_L1_data = L1_struct(trX, trY)
te_L1_data = L1_struct(teX, teY)


X = tf.placeholder("float", [None, None, input_vec_size])
Y = tf.placeholder("float", [None, 2])
sequence_length = tf.placeholder(tf.int32, [None])


def model(X):
    lstm = rnn.LSTMCell(hiede_size, cell_clip=100000,
                        # activation=tf.nn.relu,
                        forget_bias=1.0, state_is_tuple=True)
    # mlstm_cell = rnn.MultiRNNCell([lstm  for _ in range(3)], state_is_tuple=True)
    # print(X.shape)
    init_state = lstm.zero_state(20, dtype=tf.float32)
    outputs, last_states = tf.nn.dynamic_rnn(
        cell=lstm,
        dtype=tf.float32,
        sequence_length=sequence_length,
        initial_state=init_state,
        inputs=X)
    # return last_states.h, last_states.c  # State size to initialize the stat
    return last_states.h, last_states.c  # State size to initialize the stat


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


w_1 = init_weights([hiede_size, 64])
b_1 = init_weights([64])
w_2 = init_weights([64, 64])
b_2 = init_weights([64])
w_3 = init_weights([64, 32])
b_3 = init_weights([32])
w_out = init_weights([32, 2])
b_out = init_weights([2])

cnn_output, states = model(X)
h1 = tf.nn.tanh(tf.matmul(cnn_output, w_1) + b_1)
h2 = tf.nn.tanh(tf.matmul(h1, w_2) + b_2)
h3 = tf.nn.tanh(tf.matmul(h2, w_3) + b_3)
h4 = tf.matmul(h3, w_out) + b_out
py_x = tf.nn.softmax(h4)
cost = -tf.reduce_sum(Y*tf.log(tf.clip_by_value(py_x, 1e-5, 1)))
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
        for t in range(50):
            batch_data, batch_label, seq = tr_L1_data.batch(20)
            sess.run(train_op, feed_dict={X: batch_data, Y: batch_label, sequence_length: seq})
            # temp = sess.run([h4,h1,cnn_output,py_x], feed_dict={X: batch_data, Y: batch_label, sequence_length: seq})
            # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # print(temp[3][0])
            # print(temp[0][0])
            # print(temp[1][0])
            # print(temp[2][0])
            if t % 10 == 0:
                print('train_acc', sess.run(accuracy, feed_dict={X: batch_data, Y: batch_label, sequence_length: seq}))

        batch_data, batch_label, seq = tr_L1_data.batch(20)
        # print('last_states', sess.run(states, feed_dict={X: batch_data, Y: batch_label, sequence_length: seq}))
        print(i, sess.run(accuracy, feed_dict={X: batch_data, Y: batch_label, sequence_length: seq}))
        p = sess.run(py_x, feed_dict={X: batch_data, Y: batch_label, sequence_length: seq})
        p = np.concatenate((p, batch_label), 1)
        print(p)