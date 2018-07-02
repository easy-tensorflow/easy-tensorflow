import tensorflow as tf
import numpy as np
from ops import *


input_dim = 1
max_time = 4
num_hidden_units = 10
out_dim = 1

x = tf.placeholder(tf.float32, [None, max_time, input_dim])
y = tf.placeholder(tf.float32, [None])

# create weight matrix initialized randomely from N~(0, 0.01)
W = weight_variable(shape=[num_hidden_units, out_dim])

# create bias vector initialized as zero
b = bias_variable(shape=[out_dim])

# pred_out = Dynamic_LSTM(x, W, b, num_hidden_units)
pred_out = Static_LSTM(x, W, b, max_time, num_hidden_units)

cost = tf.reduce_mean(tf.square(pred_out - y))
train_op = tf.train.AdamOptimizer().minimize(cost)

# Creating the op for initializing all variables
init = tf.global_variables_initializer()

x_train = np.array([[[1], [2], [5], [6]],
           [[5], [7], [7], [8]],
           [[3], [4], [5], [9]]])
y_train = np.array([[1, 3, 8, 14],
           [5, 12, 19, 27],
           [3, 7, 12, 21]])

x_test = np.array([[[1], [2], [3], [4]],
          [[4], [5], [6], [7]]])

y_test = np.array([[[1], [3], [6], [10]],
            [[4], [9], [15], [22]]])


with tf.Session() as sess:
    sess.run(init)
    for i in range(10000):
        _, mse = sess.run([train_op, cost], feed_dict={x: x_train, y: y_train})
        if i % 1000 == 0:
            print('Step {}, MSE={}'.format(i, mse))
    # Test
        y_pred = sess.run(pred_out, feed_dict={x: x_test})

    for i, x in enumerate(x_test):
        print("When the input is {}".format(x))
        print("The ground truth output should be {}".format(y_test[i]))
        print("And the model thinks it is {}\n".format(y_pred[i]))
