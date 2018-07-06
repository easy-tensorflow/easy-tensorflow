import tensorflow as tf


# weight and bais wrappers
def weight_variable(shape):
    """
    Create a weight variable with appropriate initialization
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
    :param shape: bias variable shape
    :return: initialized bias variable
    """
    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b',
                           dtype=tf.float32,
                           initializer=initial)


def RNN(x, weights, biases, num_hidden):
    """
    :param x: inputs of size [batch_size, max_time, input_dim]
    :param weights: matrix of fully-connected output layer weights
    :param biases: vector of fully-connected output layer biases
    :param num_hidden: number of hidden units
    """
    cell = tf.nn.rnn_cell.BasicRNNCell(num_hidden)
    outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    out = tf.matmul(outputs[:, -1, :], weights) + biases
    return out
