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


def RNN(x, weights, biases, n_hidden, seq_max_len, seq_len):
    """
    :param x: inputs of shape [batch_size, max_time, input_dim]
    :param weights: matrix of fully-connected output layer weights
    :param biases: vector of fully-connected output layer biases
    :param n_hidden: number of hidden units
    :param seq_max_len: sequence maximum length
    :param seq_len: length of each sequence of shape [batch_size,]
    """
    cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    outputs, states = tf.nn.dynamic_rnn(cell, x, sequence_length=seq_len, dtype=tf.float32)

    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * seq_max_len + (seq_len - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)
    out = tf.matmul(outputs, weights) + biases
    return out


def LSTM(x, weights, biases, n_hidden, seq_max_len, seq_len):
    """
    :param x: inputs of shape [batch_size, max_time, input_dim]
    :param weights: matrix of fully-connected output layer weights
    :param biases: vector of fully-connected output layer biases
    :param n_hidden: number of hidden units
    :param seq_max_len: sequence maximum length
    :param seq_len: length of each sequence of shape [batch_size,]
    """
    cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
    outputs, states = tf.nn.dynamic_rnn(cell, x, sequence_length=seq_len, dtype=tf.float32)

    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * seq_max_len + (seq_len - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)
    out = tf.matmul(outputs, weights) + biases
    return out
