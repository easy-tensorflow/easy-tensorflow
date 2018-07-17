import tensorflow as tf
import numpy as np
from ops import *
from utils import next_batch, generate_data

# Data Dimensions
input_dim = 1           # input dimension
seq_max_len = 4         # sequence maximum length
out_dim = 1             # output dimension

# Parameters
learning_rate = 0.01    # The optimization initial learning rate
training_steps = 10000  # Total number of training steps
batch_size = 10         # batch size
display_freq = 1000     # Frequency of displaying the training results

# Network Configuration
num_hidden_units = 10   # number of hidden units

# Create the graph for the model
# Placeholders for inputs(x), input sequence lengths (seqLen) and outputs(y)
x = tf.placeholder(tf.float32, [None, seq_max_len, input_dim])
seqLen = tf.placeholder(tf.int32, [None])
y = tf.placeholder(tf.float32, [None, 1])

# create weight matrix initialized randomly from N~(0, 0.01)
W = weight_variable(shape=[num_hidden_units, out_dim])

# create bias vector initialized as zero
b = bias_variable(shape=[out_dim])

# Network predictions
pred_out = RNN(x, W, b, num_hidden_units, seq_max_len, seqLen)
# pred_out = LSTM(x, W, b, num_hidden_units, seq_max_len, seqLen)

# Define the loss function (i.e. mean-squared error loss) and optimizer
cost = tf.reduce_mean(tf.square(pred_out - y))
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Creating the op for initializing all variables
init = tf.global_variables_initializer()

# ==========
#  TOY DATA
# ==========
x_train, y_train, seq_len_train = generate_data(count=1000, max_length=seq_max_len, dim=input_dim)
x_test, y_test, seq_len_test = generate_data(count=5, max_length=seq_max_len, dim=input_dim)

# x_test = np.array([[[1], [2], [3], [4]],
#                     [[1], [2], [0], [0]],
#                    [[4], [5], [3], [9]]])
# seq_len_test = np.array([4, 2, 4])
# y_test = np.array([[10], [3], [21]])
# ==========

# Launch the graph (session)
with tf.Session() as sess:
    sess.run(init)
    print('----------Training---------')
    for i in range(training_steps):
        x_batch, y_batch, seq_len_batch = next_batch(x_train, y_train, seq_len_train, batch_size)
        _, mse = sess.run([train_op, cost], feed_dict={x: x_batch, y: y_batch, seqLen: seq_len_batch})
        if i % display_freq == 0:
            print('Step {0:<6}, MSE={1:.4f}'.format(i, mse))
    # Test
    y_pred = sess.run(pred_out, feed_dict={x: x_test, seqLen: seq_len_test})
    print('--------Test Results-------')
    for i, x in enumerate(y_test):
        print("When the ground truth output is {}, the model thinks it is {}"
              .format(y_test[i], y_pred[i]))
