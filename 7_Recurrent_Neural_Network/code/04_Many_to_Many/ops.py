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


def LSTM(x, weights, biases, num_hidden, x_len):
    """
    :param x: inputs of size [T, batch_size, input_size]
    :param weights: matrix of fully-connected output layer weights
    :param biases: vector of fully-connected output layer biases
    :param num_hidden: number of hidden units
    """
    cell = tf.nn.rnn_cell.LSTMCell(num_hidden)
    outputs, states = tf.nn.dynamic_rnn(cell, x, sequence_length=x_len, dtype=tf.float32)
    num_examples = tf.shape(x)[0]
    w_repeated = tf.tile(tf.expand_dims(weights, 0), [num_examples, 1, 1])
    out = tf.matmul(outputs, w_repeated) + biases
    out = tf.squeeze(out)
    return out
