# imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

from ops import fc_layer
from utils import plot_max_active, plot_images

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print("Size of:")
print("- Training-set:\t\t{}".format(len(mnist.train.labels)))
print("- Test-set:\t\t{}".format(len(mnist.test.labels)))
print("- Validation-set:\t{}".format(len(mnist.validation.labels)))

# hyper-parameters
logs_path = "./logs/full"  # path to the folder that we want to save the logs for Tensorboard
learning_rate = 0.001  # The optimization learning rate
epochs = 10  # Total number of training epochs
batch_size = 100  # Training batch size
display_freq = 100  # Frequency of displaying the training results

# Network Parameters
# We know that MNIST images are 28 pixels in each dimension.
img_h = img_w = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_h * img_w

# number of units in the hidden layer
h1 = 100

# level of the noise in noisy data
noise_level = 0.6

# Create graph
# Placeholders for inputs (x), outputs(y)
with tf.variable_scope('Input'):
    x_original = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='X_original')
    x_noisy = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='X_noisy')


fc1, W1 = fc_layer(x_noisy, h1, 'Hidden_layer', use_relu=True)
out, W2 = fc_layer(fc1, img_size_flat, 'Output_layer', use_relu=False)

# calculate the activation
h_active = W1 / tf.sqrt(tf.reduce_sum(tf.square(W1), axis=0)) # [784, 100]

# Define the loss function, optimizer, and accuracy
with tf.variable_scope('Train'):
    with tf.variable_scope('Loss'):
        loss = tf.reduce_mean(tf.losses.mean_squared_error(x_original, out), name='loss')
        tf.summary.scalar('loss', loss)
    with tf.variable_scope('Optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()

# Add 5 images from original, noisy and reconstructed samples to summaries
tf.summary.image('original', tf.reshape(x_original, (-1, img_w, img_h, 1)), max_outputs=5)
tf.summary.image('noisy', tf.reshape(x_noisy, (-1, img_w, img_h, 1)), max_outputs=5)
tf.summary.image('reconstructed', tf.reshape(out, (-1, img_w, img_h, 1)), max_outputs=5)

merged = tf.summary.merge_all()

# Launch the graph (session)
with tf.Session() as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter(logs_path, sess.graph)
    num_tr_iter = int(mnist.train.num_examples / batch_size)
    global_step = 0
    for epoch in range(epochs):
        print('Training epoch: {}'.format(epoch + 1))
        for iteration in range(num_tr_iter):
            batch_x, _ = mnist.train.next_batch(batch_size)
            batch_x_noisy = batch_x + noise_level * np.random.normal(loc=0.0, scale=1.0, size=batch_x.shape)

            global_step += 1
            # Run optimization op (backprop)
            feed_dict_batch = {x_original: batch_x, x_noisy: batch_x_noisy}
            _, summary_tr = sess.run([optimizer, merged], feed_dict=feed_dict_batch)
            train_writer.add_summary(summary_tr, global_step)

            if iteration % display_freq == 0:
                # Calculate and display the batch loss and accuracy
                loss_batch = sess.run(loss,
                                      feed_dict=feed_dict_batch)
                print("iter {0:3d}:\t Reconstruction loss={1:.3f}".
                      format(iteration, loss_batch))

        # Run validation after every epoch
        x_valid_original  = mnist.validation.images
        x_valid_noisy = x_valid_original + noise_level * np.random.normal(loc=0.0, scale=1.0, size=x_valid_original.shape)

        feed_dict_valid = {x_original: x_valid_original, x_noisy: x_valid_noisy}
        loss_valid = sess.run(loss, feed_dict=feed_dict_valid)
        print('---------------------------------------------------------')
        print("Epoch: {0}, validation loss: {1:.3f}".
              format(epoch + 1, loss_valid))
        print('---------------------------------------------------------')

    # Test the network after training
    # Make a noisy image
    test_samples = 5
    x_test = mnist.test.images[:test_samples]
    x_test_noisy = x_test + noise_level * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
    # Reconstruct a clean image from noisy image
    x_reconstruct = sess.run(out, feed_dict={x_noisy: x_test_noisy})
    # Calculate the loss between reconstructed image and original image
    loss_test = sess.run(loss, feed_dict={x_original: x_test, x_noisy: x_test_noisy})
    print('---------------------------------------------------------')
    print("Test loss of original image compared to reconstructed image : {0:.3f}".format(loss_test))
    print('---------------------------------------------------------')

    # Plot original image, noisy image and reconstructed image
    plot_images(x_test, x_test_noisy, x_reconstruct)

    # Plot the images that maximally activate the hidden units
    plot_max_active(sess.run(h_active))
    plt.show()





# Load the test set
x_test = mnist.test.images
y_test = mnist.test.labels


# Initialize the embedding variable with the shape of our desired tensor
tensor_shape = (x_test.shape[0] , fc1.get_shape()[1].value) # [test_set , h1] = [10000 , 200]
embedding_var = tf.Variable(tf.zeros(tensor_shape),
                            name='fc1_embedding')
# assign the tensor that we want to visualize to the embedding variable
embedding_assign = embedding_var.assign(fc1)

