# imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# hyper-parameters
learning_rate = 0.001
epochs = 10
batch_size = 128
display_step = 100

# Network Parameters
d_input = 784  # MNIST data input dimension (img shape: 28*28)
h1 = 200  # number of units in hidden layer
n_classes = 10  # MNIST total classes (0-9 digits)


# weight and bais wrappers
def weight_variable(name, shape):
    """
    Create a weight variable with appropriate initialization
    name: weight name
    shape: weight shape
    
    return: initialized weight variable
    """
    initer = tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable('W_' + name,
                           dtype=tf.float32,
                           shape=shape,
                           initializer=initer)


def bias_variable(name, shape):
    """
    Create a bias variable with appropriate initialization
    name: bias variable name
    shape: bias variable shape
    
    return: initialized bias variable
    """
    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b_' + name,
                           dtype=tf.float32,
                           initializer=initial)


def fc_layer(x, num_nodes, name, use_relu=True):
    """
    Creates a fully-connected layer
    :param x: input from previous layer
    :param num_nodes: number of hidden units in the fully-connected layer
    :param name: layer name
    :param use_relu: boolean to add ReLU non-linearity (or not)
    
    :return: The output array
    """
    in_dim = x.get_shape()[1]
    W = weight_variable(name, shape=[in_dim, num_nodes])
    b = bias_variable(name, [num_nodes])
    layer = tf.matmul(x, W)
    layer += b
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


# Create graph
# Placeholders for inputs (x), outputs(y)
x = tf.placeholder(tf.float32, shape=[None, d_input], name='X')
y = tf.placeholder(tf.float32, shape=[None, n_classes], name='Y')
fc1 = fc_layer(x, h1, 'FC1', use_relu=True)
output_logits = fc_layer(fc1, n_classes, 'OUT', use_relu=False)

# Define loss, optimizer, accuracy for training
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output_logits))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
correct_prediction = tf.equal(tf.argmax(output_logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# predict classes for testing and evluation
predicted_classes = tf.argmax(output_logits, axis=1)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph (session)
with tf.Session() as sess:
    sess.run(init)
    num_tr_iter = int(mnist.train.num_examples / batch_size)
    for epoch in range(epochs):
        print('Training epoch: {}'.format(epoch))
        for iter in range(num_tr_iter):
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            # Run optimization op (backprop)
            _, loss_batch, acc = sess.run([optimizer, loss, accuracy],
                                          feed_dict={x: batch_x, y: batch_y})

            if iter % display_step == 0:
                # Calculate batch loss and accuracy
                loss_batch, acc_batch = sess.run([loss, accuracy],
                                                 feed_dict={x: batch_x, y: batch_y})
                print("iter {0:3d}:\t Loss={1:.2f},\tTraining Accuracy={2:.2f}".
                      format(iter, loss_batch, acc_batch))

        # validation
        loss_valid, acc_valid = sess.run([loss, accuracy],
                                         feed_dict={x: mnist.validation.images,
                                                    y: mnist.validation.labels})
        print("Validation Accuracy:",
              sess.run(accuracy, feed_dict={x: mnist.validation.images,
                                            y: mnist.validation.labels}))
    # test the network
    # Calculate accuracy
    print("Testing Accuracy:",
          sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y: mnist.test.labels}))
    # Predict single images
    n_images = 4
    test_images = mnist.test.images[:n_images]
    preds = sess.run(predicted_classes, feed_dict={x: test_images})
    # Display
    fig = plt.figure()
    for i in range(n_images):
        ax = fig.add_subplot(2, 2, i + 1)
        ax.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray')
        ax.set_title("Model prediction: {}".format(preds[i]))
        ax.axis('off')
    plt.show()
