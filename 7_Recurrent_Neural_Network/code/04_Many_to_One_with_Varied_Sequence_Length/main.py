import tensorflow as tf
import numpy as np
from ops import *

# Data Dimensions
input_dim = 1  # input dimension
seq_max_len = 4  # sequence maximum length
out_dim = 1  # output dimension

# Parameters
learning_rate = 0.01  # The optimization initial learning rate
training_steps = 4000  # Total number of training steps
batch_size = 2  # batch size
display_freq = 1000  # Frequency of displaying the training results

# Network Configuration
num_hidden_units = 10  # number of hidden units

# Create the graph for the model
# Placeholders for inputs(x), input sequence lengths (seqLen) and outputs(y)
x = tf.placeholder(tf.float32, [None, seq_max_len, input_dim])
seqLen = tf.placeholder(tf.int32, [None])
y = tf.placeholder(tf.float32, [None, 1])

# create weight matrix initialized randomely from N~(0, 0.01)
W = weight_variable(shape=[num_hidden_units, out_dim])

# create bias vector initialized as zero
b = bias_variable(shape=[out_dim])

# Network predictions
pred_out = LSTM(x, W, b, num_hidden_units, seq_max_len, seqLen)

# Define the loss function (i.e. mean-squared error loss) and optimizer
cost = tf.reduce_mean(tf.square(pred_out - y))
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Creating the op for initializing all variables
init = tf.global_variables_initializer()

# ==========
#  TOY DATA
# ==========
x_train = np.array([[[1], [2], [5], [6]],
                    [[5], [7], [4], [0]],
                    [[2], [5], [1], [3]],
                    [[9], [0], [3], [1]]])
seq_len_train = np.array([4, 3, 4, 4])
y_train = np.array([[14], [16], [11], [13]])

x_test = np.array([[[1], [2], [3], [4]],
                   [[4], [5], [3], [9]]])

y_test = [[10], [21]]

# Launch the graph (session)
with tf.Session() as sess:
    sess.run(init)
    for i in range(training_steps):
        perm = np.random.permutation(len(x_train))
        x_batch = x_train[perm][:batch_size]
        x_len = seq_len_train[perm][:batch_size]
        y_batch = y_train[perm][:batch_size]
        _, mse = sess.run([train_op, cost], feed_dict={x: x_batch,
                                                       y: y_batch,
                                                       seqLen: x_len})
        if i % display_freq == 0:
            print('Step {}, MSE={}'.format(i, mse))
    # Test the model when training is done
    y_pred = sess.run(pred_out, feed_dict={x: x_test,
                                           seqLen: [4, 1]})

    print(y_pred)
