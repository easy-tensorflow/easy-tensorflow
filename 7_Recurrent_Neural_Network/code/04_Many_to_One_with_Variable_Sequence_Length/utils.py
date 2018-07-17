import numpy as np


def generate_data(count=1000, max_length=4, dim=1):
    x = np.random.randint(0, 10, size=(count, max_length, dim))
    length = np.random.randint(1, max_length+1, count)
    for i in range(count):
        x[i, length[i]:, :] = 0
    y = np.sum(x, axis=1)
    return x, y, length


def next_batch(x, y, seq_len, batch_size):
    N = x.shape[0]
    batch_indeces = np.random.permutation(N)[:batch_size]
    x_batch = x[batch_indeces]
    y_batch = y[batch_indeces]
    seq_len_batch = seq_len[batch_indeces]
    return x_batch, y_batch, seq_len_batch
