    import tensorflow as tf
import numpy as np

# Create input data
# Lets say you have a batch of two examples, one is of length 10, and the other of length 6.
# Each one is a vector of 8 numbers (time-steps=8)

X = np.random.randn(2, 10, 8)   # [batch_size, max_time (or length), input_dim]

# The second example is of length 6
X[1, 6:] = 0
X_lengths = [10, 6]

cell = tf.nn.rnn_cell.LSTMCell(num_units=100)
outputs, last_states = tf.nn.dynamic_rnn(cell, X, sequence_length=X_lengths, dtype=tf.float64)
# X: input of shape [batch_size, max_time, input_dim]
# sequence_length: (optional) An int32/int64 vector sized `[batch_size]`. Used to copy-through
# state and zero-out outputs when past a batch element's sequence length.  So it's more for
# correctness than performance.
# dtype: (optional) The data type for the initial state and expected output.
#  (If there is no initial_state, you must give a dtype.)

# outputs: [batch_size, max_time, num_hidden_units] (=ht)
# last_states: the last state for each example (cT, hT), each of shape [batch_size, num_hidden_units]
# If cells are `LSTMCells`, `state` will be a tuple containing a `LSTMStateTuple` for each cell.

print()

