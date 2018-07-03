import tensorflow as tf
import numpy as np
from ops import *

input_dim = 1
max_time = 4
num_hidden_units = 10
out_dim = 1

x = tf.placeholder(tf.float32, [None, max_time, input_dim])
x_len = tf.placeholder(tf.int32, [None])
y = tf.placeholder(tf.float32, [None, 1])

# create weight matrix initialized randomely from N~(0, 0.01)
W = weight_variable(shape=[num_hidden_units, out_dim])

# create bias vector initialized as zero
b = bias_variable(shape=[out_dim])

pred_out = LSTM(x, W, b, num_hidden_units)

cost = tf.reduce_mean(tf.square(pred_out - y))
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

# Creating the op for initializing all variables
init = tf.global_variables_initializer()

x_train = [[[1], [2], [5], [6]],
           [[5], [7], [4], [0]],
           [[2], [5], [1], [3]],
           [[9], [0], [3], [1]]]
y_train = [[14], [16], [11], [13]]

x_test = [[[1], [2], [3], [4]],
          [[4], [5], [6], [7]]]

y_test = [[10], [22]]

with tf.Session() as sess:
    sess.run(init)
    for i in range(10000):
        _, mse = sess.run([train_op, cost], feed_dict={x: x_train, y: y_train, x_len: [4, 3, 4, 4]})
        if i % 1000 == 0:
            print('Step {}, MSE={}'.format(i, mse))
    # Test
    y_pred = sess.run(pred_out, feed_dict={x: x_test, x_len:[4, 4]})
