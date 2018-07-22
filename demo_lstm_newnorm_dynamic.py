# -*- coding: utf-8 -*-
from utils import *

input_vec_size = lstm_size = 25  # 输入向量的维度
input_time_mins = 30
pre_time_mins = 10
time_step_size = input_time_mins * 20  # 循环层长度
batch_size = 50

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
    def __init__(self, data, label, filter_label):
        self.data = data
        self.label = label
        self.filter_label = filter_label

    def batch(self, size):
        data_len = len(self.data)
        index = np.array(range(data_len))
        random.shuffle(index)
        data_batch_loc = index[: size]
        batch_data = self.data[data_batch_loc]
        batch_label = self.label[data_batch_loc]
        batch_filter_label = self.filter_label[data_batch_loc]
        batch_fix_data = []
        batch_fix_label = []
        sequence_length = []

        for batch in range(len(batch_data)):

            temp_len = len(batch_data[batch])
            p = np.random.random()
            temp_index = np.random.randint(low=200, high=temp_len)
            if p > 0.51:
                p_index = [idx for (idx, val) in enumerate(batch_filter_label[batch]) if list(val) != [1, 0] and idx>200]
                if p_index != []:
                    temp_index = p_index[np.random.randint(low=0, high=len(p_index))]+1

            copy_data = copy.deepcopy(batch_data[batch][: temp_index, :])
            sequence_length.append(len(copy_data))
            mean_price = np.mean(copy_data[:200, 0])
            std_price = np.std(copy_data[:200, 0])
            mean_volume = np.mean(copy_data[:200, 1])
            std_volume = np.std(copy_data[:200, 1])
            mean_ba = np.mean(copy_data[:200, 15:25])
            std_ba = np.std(copy_data[:200, 15:25])
            copy_data[:, 0] = (copy_data[:, 0] - mean_price) / mean_price
            copy_data[:, 1] = (copy_data[:, 1] - mean_volume) / std_volume
            for i in range(5, 15):
                copy_data[:, i] = (copy_data[:, i] - mean_price) / mean_price
            for i in range(15, 25):
                copy_data[:, i] = (copy_data[:, i] - mean_ba) / std_ba
            batch_fix_data.append(copy_data)
            batch_fix_label.append(batch_filter_label[batch][temp_index-1])
            # print("BATCH %d DONE", (batch))

            # 可视化检查
            # fig = plt.figure()
            # ax1 = fig.add_subplot(422)
            # ax1.plot(batch_data[batch][:,0])
            # buy_index = []
            # sell_index = []
            # for i in range(len(batch_label[batch])):
            #     if list(batch_filter_label[batch][i]) == [0, 1, 0]:
            #         buy_index.append(i)
            #     if list(batch_filter_label[batch][i]) == [0, 0, 1]:
            #         sell_index.append(i)
            # ax1.scatter(buy_index, batch_data[batch][buy_index, 0], c='r')
            # ax1.scatter(sell_index, batch_data[batch][sell_index, 0], c='y')
            # ax1.axvline(temp_index-1)
            #
            # ax2 = fig.add_subplot(421)
            # ax2.plot(batch_data[batch][:, 0])
            # buy_index = []
            # sell_index = []
            # for i in range(len(batch_label[batch])):
            #     if list(batch_label[batch][i]) == [0, 1, 0]:
            #         buy_index.append(i)
            #     if list(batch_label[batch][i]) == [0, 0, 1]:
            #         sell_index.append(i)
            # ax2.scatter(buy_index, batch_data[batch][buy_index, 0], c='r')
            # ax2.scatter(sell_index, batch_data[batch][sell_index, 0], c='y')
            # ax2.axvline(temp_index - 1)
            # ax2.set_title(str(batch_filter_label[batch][temp_index-1]))
            #
            # ax3 = fig.add_subplot(423)
            # ax3.plot(copy_data[:, 0])
            # ax4 = fig.add_subplot(424)
            # ax4.plot(copy_data[:, 1])
            # ax5 = fig.add_subplot(425)
            # ax5.plot(copy_data[:, 5:10])
            # ax6 = fig.add_subplot(426)
            # ax6.plot(copy_data[:, 10:15])
            # ax7 = fig.add_subplot(427)
            # ax7.plot(copy_data[:, 15:20])
            # ax8 = fig.add_subplot(428)
            # ax8.plot(copy_data[:, 20:25])
            # plt.show()
        batch_fix_data = np.array(batch_fix_data)
        batch_fix_data = apply_to_zeros(batch_fix_data)
        return batch_fix_data, np.array(batch_fix_label), np.array(sequence_length)





def mat2list(data, keys, use_datalen):
    time_s = np.array([i[0] for i in data['Time']])
    temp = np.array([0 for i in range(len(time_s))])
    temp[time_s > 94500000] = 1
    day_b = []
    for i in range(len(time_s)):
        if temp[i]==0 and temp[i+1]==1:
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
    return list_data, day_b[:use_datalen], day_e[:use_datalen]


def list2nparray(list_data):
    # 生成数据矩阵，部分数据one-hot化处理
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
    return  np_data


# 载入数据进行部分预处理
data = sio.loadmat('SZ300112.mat')
keys = ['Price', 'Volume', 'BSFlag', 'AskPrice5', 'BidPrice5', 'AskVolume5', 'BidVolume5']
list_data, day_b, day_e = mat2list(data, keys, 100)
del data
np_data = list2nparray(list_data)
del list_data

all_data = []
all_label = []
all_filter_label = []
for day in range(len(day_b)):
    # print(np_data[day_b[day]:day_e[day]][0])
    day_data, day_label= MMdata2label(np_data[day_b[day]:day_e[day]], pre_time_mins)
    filter_label = day_label
    all_data.append(day_data)
    all_label.append(day_label)
    all_filter_label.append(filter_label)
all_data = np.array(all_data)
all_label = np.array(all_label)
all_filter_label = np.array(all_filter_label)
all_data_size = len(all_data)
tr_size = int(all_data_size * 0.7)
te_size = all_data_size - tr_size
data_index = np.arange(all_data_size)
np.random.shuffle(data_index)
train_indices = data_index[:tr_size]
trX = all_data[train_indices]
trY = all_label[train_indices]
filter_trY = all_filter_label[train_indices]
test_indices = data_index[tr_size:]
teX = all_data[test_indices]
teY = all_label[test_indices]
filter_teY = all_filter_label[test_indices]


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X):
    # X, input shape: (batch_size, time_step_size, input_vec_size)
    # XT shape: (time_step_size, batch_size, input_vec_size)
    XT = tf.transpose(X, [1, 0, 2])  # permute time_step_size and batch_size,[28, 128, 28]
    # XR shape: (time_step_size * batch_size, input_vec_size)
    XR = tf.reshape(XT, [-1, lstm_size])  # each row has input for each lstm cell (lstm_size=input_vec_size)
    # Each array shape: (batch_size, input_vec_size)
    X_split = tf.split(XR, time_step_size,0)  # split them to time_step_size (28 arrays),shape = [(128, 28),(128, 28)...]
    # Make lstm with lstm_size (each input vector size). num_units=lstm_size; forget_bias=1.0
    lstm = rnn.BasicLSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=True)
    outputs, last_states = tf.nn.dynamic_rnn(
        cell=lstm,
        dtype=tf.float32,
        sequence_length=sequence_length,
        inputs=X)
    return last_states.h, last_states.c  # State size to initialize the stat


tr_L1_data = L1_struct(trX, trY, filter_trY)
te_L1_data = L1_struct(teX, teY, filter_teY)

X = tf.placeholder("float", [None, None, lstm_size])
Y = tf.placeholder("float", [None, 2])
sequence_length = tf.placeholder(tf.int32, [None])

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
        for t in range(100):
            batch_data, batch_label, seq = tr_L1_data.batch(50)
            sess.run(train_op, feed_dict={X: batch_data, Y: batch_label, sequence_length: seq})
            if t%10 == 0:
                print('train_acc', sess.run(accuracy, feed_dict={X: batch_data, Y: batch_label, sequence_length: seq}))

        batch_data, batch_label, seq= te_L1_data.batch(50)
        print('last_states', sess.run(states, feed_dict={X: batch_data, Y: batch_label, sequence_length: seq}))
        print(i, sess.run(accuracy, feed_dict={X: batch_data, Y: batch_label, sequence_length: seq}))
        p = sess.run(py_x, feed_dict={X: batch_data, Y: batch_label, sequence_length: seq})
        # print(batch_data)
        p = np.concatenate((p, batch_label), 1)
        print(p)