# imports
import tensorflow as tf
import numpy as np
from ops import fc_layer
from utils import plot_example_errors

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print("Size of:")
print("- Training-set:\t\t{}".format(len(mnist.train.labels)))
print("- Test-set:\t\t{}".format(len(mnist.test.labels)))
print("- Validation-set:\t{}".format(len(mnist.validation.labels)))

# hyper-parameters
learning_rate = 0.001  # The optimization learning rate
epochs = 10  # Total number of training epochs
batch_size = 100  # Training batch size
display_freq = 100  # Frequency of displaying the training results

# Network Parameters
# We know that MNIST images are 28 pixels in each dimension.
img_h = img_w = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_h * img_w

# Number of classes, one class for each of 10 digits.
n_classes = 10

# number of units in the first hidden layer
h1 = 200

# Create graph
# Placeholders for inputs (x), outputs(y)
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='X')
y = tf.placeholder(tf.float32, shape=[None, n_classes], name='Y')
fc1 = fc_layer(x, h1, 'FC1', use_relu=True)
output_logits = fc_layer(fc1, n_classes, 'OUT', use_relu=False)

# Define the loss function, optimizer, and accuracy
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output_logits), name='loss')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(loss)
correct_prediction = tf.equal(tf.argmax(output_logits, 1), tf.argmax(y, 1), name='correct_pred')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

# Network predictions
cls_prediction = tf.argmax(output_logits, axis=1, name='predictions')

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph (session)
with tf.Session() as sess:
    sess.run(init)
    num_tr_iter = int(mnist.train.num_examples / batch_size)
    for epoch in range(epochs):
        print('Training epoch: {}'.format(epoch))
        for iteration in range(num_tr_iter):
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            # Run optimization op (backprop)
            feed_dict_batch = {x: batch_x, y: batch_y}
            sess.run(optimizer, feed_dict=feed_dict_batch)

            if iteration % display_freq == 0:
                # Calculate and display the batch loss and accuracy
                loss_batch, acc_batch = sess.run([loss, accuracy],
                                                 feed_dict=feed_dict_batch)
                print("iter {0:3d}:\t Loss={1:.2f},\tTraining Accuracy={2:.01%}".
                      format(iteration, loss_batch, acc_batch))

        # Run validation after every epoch
        feed_dict_valid = {x: mnist.validation.images, y: mnist.validation.labels}
        loss_valid, acc_valid = sess.run([loss, accuracy], feed_dict=feed_dict_valid)
        print('---------------------------------------------------------')
        print("Epoch: {0}, validation loss: {1:.2f}, validation accuracy: {2:.01%}".
              format(epoch + 1, loss_valid, acc_valid))
        print('---------------------------------------------------------')

    # Test the network after training
    feed_dict_test = {x: mnist.test.images, y: mnist.test.labels}
    loss_test, acc_test = sess.run([loss, accuracy], feed_dict=feed_dict_test)
    print("Test loss: {0:.2f}, test accuracy: {1:.01%}".format(loss_test, acc_test))

    # Plot some of the misclassified examples
    cls_pred = sess.run(cls_prediction, feed_dict=feed_dict_test)
    cls_true = np.argmax(mnist.test.labels, axis=1)
    plot_example_errors(mnist.test.images, cls_true, cls_pred)

