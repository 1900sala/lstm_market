import tensorflow as tf
import numpy as np

shape=tf.placeholder(tf.float32, shape=[None, 227,227,3] )

def apply_to_zeros(lst):
    inner_max_len = max(map(len, lst))
    result = np.zeros([len(lst), inner_max_len, len(lst[0][0])])
    for i, row in enumerate(lst):
        for j, val in enumerate(row):
            for k, temp in enumerate(val):
                result[i][j][k] = temp
    return result
# 产生2个batch数据，句子length为3，embedding大小为4
# X = np.random.randn(2, 3, 4)
X = np.array([[[-0.0237397,   0.22122294 , 0.08335812, -1.61402272],
  [ 0.40904425,  0.78221612,  1.53902757, -0.95241871],
  [ 0.91778039,-1.50089013, -0.3505784  , 0.22788677]],

 [[ 1.23179831,  0.88142978, -0.1662979 , -0.23657662],
  [-1.5829275,   1.42711468, -1.64400619 ,-1.63445493]]])


# 第二个batch长度为2
# X[1,2:] = 0
X_lengths = [3, 2]
X = apply_to_zeros(X)
print(X)

# cell = tf.contrib.rnn.BasicLSTMCell(num_units=64, state_is_tuple=True)
cell = tf.contrib.rnn.BasicLSTMCell(num_units=5,state_is_tuple=True)

outputs, last_states = tf.nn.dynamic_rnn(
    cell=cell,
    dtype=tf.float64,
    # sequence_length=X_lengths,
    inputs=X)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    o=sess.run(outputs)
    s=sess.run(last_states)
    print('output\n',o)
    print('last_o\n',o[:,-1,:])# 从output中取最后一次输出
    #
    # print('--------------------')
    # print('s\n',s)
    print('s.c\n',s.c)    # 这是门控单元的权重，这里不需要
    print('s.h\n',s.h)    #s.h就是最后一次输出的状态