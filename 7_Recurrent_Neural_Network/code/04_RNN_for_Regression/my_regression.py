import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np

input_dim = 1
seq_size = 4
num_hidden_units = 10
out_dim = 1


# weight and bais wrappers
def weight_variable(shape):
    """
    Create a weight variable with appropriate initialization
    :param name: weight name
    :param shape: weight shape
    :return: initialized weight variable
    """
    initer = tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable('W',
                           dtype=tf.float32,
                           shape=shape,
                           initializer=initer)


def bias_variable(shape):
    """
    Create a bias variable with appropriate initialization
    :param name: bias variable name
    :param shape: bias variable shape
    :return: initialized bias variable
    """
    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b',
                           dtype=tf.float32,
                           initializer=initial)


def LSTM(x, weights, biases, num_hidden):
    """
    :param x: inputs of size [T, batch_size, input_size]
    :param weights: matrix of fully-connected output layer weights
    :param biases: vector of fully-connected output layer biases
    """
    cell = rnn_cell.BasicLSTMCell(num_hidden)
    outputs, states = rnn.dynamic_rnn(cell, x, dtype=tf.float32)
    num_examples = tf.shape(x)[0]
    w_repeated = tf.tile(tf.expand_dims(weights, 0), [num_examples, 1, 1])
    out = tf.matmul(outputs, w_repeated) + biases
    out = tf.squeeze(out)
    return out


def LSTM2(x, weights, biases, timesteps, num_hidden):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    num_examples = tf.shape(x)[0]
    x = tf.unstack(x, timesteps, 1)

    # Define a rnn cell with tensorflow
    cell = rnn_cell.BasicLSTMCell(num_hidden)
    w_repeated = tf.tile(tf.expand_dims(weights, 0), [num_examples, 1, 1])
    # Get lstm cell output
    # If no initial_state is provided, dtype must be specified
    # If no initial cell state is provided, they will be initialized to zero
    outputs_series, current_state = rnn.static_rnn(cell, x, dtype=tf.float32)
    outputs = tf.stack(outputs_series, axis=1)
    out = tf.matmul(outputs, w_repeated) + biases
    out = tf.squeeze(out)

    # Linear activation, using rnn inner loop last output
    return out


x = tf.placeholder(tf.float32, [None, seq_size, input_dim])
y = tf.placeholder(tf.float32, [None, seq_size])

# create weight matrix initialized randomely from N~(0, 0.01)
W = weight_variable(shape=[num_hidden_units, out_dim])

# create bias vector initialized as zero
b = bias_variable(shape=[out_dim])

# pred_out = LSTM(x, W, b, num_hidden_units)
pred_out = LSTM2(x, W, b, seq_size, num_hidden_units)

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
