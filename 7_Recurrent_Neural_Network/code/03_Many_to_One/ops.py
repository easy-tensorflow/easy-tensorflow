import tensorflow as tf


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
    cell = tf.nn.rnn_cell.LSTMCell(num_hidden)
    outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    num_examples = tf.shape(x)[0]
    w_repeated = tf.tile(tf.expand_dims(weights, 0), [num_examples, 1, 1])
    out = tf.matmul(outputs, w_repeated) + biases
    out = tf.squeeze(out)
    return out


# def Static_LSTM(x, weights, biases, timesteps, num_hidden):
#
#     # Prepare data shape to match `rnn` function requirements
#     # Current data input shape: (batch_size, timesteps, n_input)
#     # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)
#
#     # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
#     num_examples = tf.shape(x)[0]
#     x = tf.unstack(x, timesteps, 1)
#
#     # Define a rnn cell with tensorflow
#     cell = tf.nn.rnn_cell.LSTMCell(num_hidden)
#     w_repeated = tf.tile(tf.expand_dims(weights, 0), [num_examples, 1, 1])
#     # Get lstm cell output
#     # If no initial_state is provided, dtype must be specified
#     # If no initial cell state is provided, they will be initialized to zero
#     outputs_series, current_state = tf.nn.static_rnn(cell, x, dtype=tf.float32)
#     outputs = tf.stack(outputs_series, axis=1)
#     out = tf.matmul(outputs, w_repeated) + biases
#     out = tf.squeeze(out)
#
#     # Linear activation, using rnn inner loop last output
#     return out
