import tensorflow as tf
from tensorflow.contrib import rnn


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


def RNN(x, weights, biases, max_time, num_hidden):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, max_time, 1)

    # Define a rnn cell with tensorflow
    # rnn_cell = rnn.BasicRNNCell(num_hidden)
    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden)

    # Get lstm cell output
    # If no initial_state is provided, dtype must be specified
    # If no initial cell satte is provided, they will be initialized to zero
    # states_series, current_state = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    # return tf.matmul(current_state, weights) + biases
    return tf.matmul(outputs[-1], weights) + biases

