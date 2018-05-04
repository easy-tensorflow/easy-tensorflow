# imports
import os
import tensorflow as tf
import numpy as np
from ops import fc_layer
from utils import *
from tensorflow.contrib.tensorboard.plugins import projector

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("Size of:")
print("- Training-set:\t\t{}".format(len(mnist.train.labels)))
print("- Test-set:\t\t{}".format(len(mnist.test.labels)))
print("- Validation-set:\t{}".format(len(mnist.validation.labels)))

# hyper-parameters
logs_path = "./logs/full"  # path to the folder that we want to save the logs for TensorBoard
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
with tf.variable_scope('Input'):
    x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='X')
    tf.summary.image('input_image', tf.reshape(x, (-1, img_w, img_h, 1)), max_outputs=5)
    y = tf.placeholder(tf.float32, shape=[None, n_classes], name='Y')

fc1 = fc_layer(x, h1, 'Hidden_layer', use_relu=True)
output_logits = fc_layer(fc1, n_classes, 'Output_layer', use_relu=False)

# Define the loss function, optimizer, and accuracy
with tf.variable_scope('Train'):
    with tf.variable_scope('Loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output_logits), name='loss')
        tf.summary.scalar('loss', loss)
    with tf.variable_scope('Optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(loss)
    with tf.variable_scope('Accuracy'):
        correct_prediction = tf.equal(tf.argmax(output_logits, 1), tf.argmax(y, 1), name='correct_pred')
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
        tf.summary.scalar('accuracy', accuracy)
    with tf.variable_scope('Prediction'):
        # Network predictions
        cls_prediction = tf.argmax(output_logits, axis=1, name='predictions')

# Initialize the variables
init = tf.global_variables_initializer()
# Merge all summaries
merged = tf.summary.merge_all()

# Launch the graph (session)
with tf.Session() as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter(logs_path, sess.graph)
    num_tr_iter = int(mnist.train.num_examples / batch_size)
    global_step = 0
    for epoch in range(epochs + 1):
        print('Training epoch: {}'.format(epoch))
        for iteration in range(num_tr_iter):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            global_step += 1
            # Run optimization op (backprop)
            feed_dict_batch = {x: batch_x, y: batch_y}
            _, summary_tr = sess.run([optimizer, merged], feed_dict=feed_dict_batch)
            train_writer.add_summary(summary_tr, global_step)

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

    # Load the test set
    x_test = mnist.test.images
    y_test = mnist.test.labels

    # Initialize the embedding variable with the shape of our desired tensor
    tensor_shape = (x_test.shape[0], fc1.get_shape()[1].value)  # [test_set , h1] = [10000 , 200]
    embedding_var = tf.Variable(tf.zeros(tensor_shape),
                                name='fc1_embedding')
    # assign the tensor that we want to visualize to the embedding variable
    embedding_assign = embedding_var.assign(fc1)

    # Create a config object to write the configuration parameters
    config = projector.ProjectorConfig()

    # Add embedding variable
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name

    # Link this tensor to its metadata file (e.g. labels) -> we will create this file later
    embedding.metadata_path = 'metadata.tsv'

    # Specify where you find the sprite. -> we will create this image later
    embedding.sprite.image_path = 'sprite_images.png'
    embedding.sprite.single_image_dim.extend([img_w, img_h])

    # Write a projector_config.pbtxt in the logs_path.
    # TensorBoard will read this file during startup.
    projector.visualize_embeddings(train_writer, config)

    # Run session to evaluate the tensor
    x_test_fc1 = sess.run(embedding_assign, feed_dict={x: x_test})

    # Save the tensor in model.ckpt file
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(logs_path, "model.ckpt"), global_step)

    # Reshape images from vector to matrix
    x_test_images = np.reshape(np.array(x_test), (-1, img_w, img_h))
    # Reshape labels from one-hot-encode to index
    x_test_labels = np.argmax(y_test, axis=1)

    write_sprite_image(os.path.join(logs_path, 'sprite_images.png'), x_test_images)
    write_metadata(os.path.join(logs_path, 'metadata.tsv'), x_test_labels)