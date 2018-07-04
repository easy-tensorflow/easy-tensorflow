import tensorflow as tf
import numpy as np
from ops import *
from utils import next_batch

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
y = tf.placeholder(tf.float32, [None, 1])

# create weight matrix initialized randomely from N~(0, 0.01)
W = weight_variable(shape=[num_hidden_units, out_dim])

# create bias vector initialized as zero
b = bias_variable(shape=[out_dim])

# Network predictions
pred_out = LSTM(x, W, b, num_hidden_units)

# Define the loss function (i.e. mean-squared error loss) and optimizer
cost = tf.reduce_mean(tf.square(pred_out - y))
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

# Creating the op for initializing all variables
init = tf.global_variables_initializer()

# ==========
#  TOY DATA
# ==========
x_train = np.random.randint(0, 10, size=(100, 4, 1))
y_train = np.sum(x_train, axis=1)

x_test = np.random.randint(0, 10, size=(5, 4, 1))
y_test = np.sum(x_test, axis=1)
# ==========

# Launch the graph (session)
with tf.Session() as sess:
    sess.run(init)
    for i in range(training_steps):
        x_batch, y_batch = next_batch(x_train, y_train, batch_size)
        _, mse = sess.run([train_op, cost], feed_dict={x: x_batch, y: y_batch})
        if i % display_freq == 0:
            print('Step {}, MSE={}'.format(i, mse))
    # Test
    y_pred = sess.run(pred_out, feed_dict={x: x_test})

    for i, x in enumerate(y_test):
        print("When the ground truth output is {}, the model thinks it is {}"
              .format(y_test[i], y_pred[i]))
