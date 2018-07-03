import tensorflow as tf

# Load MNIST data
from ops import weight_variable, bias_variable, RNN
from utils import load_data, randomize, get_next_batch

x_train, y_train, x_valid, y_valid = load_data(mode='train')
print("Size of:")
print("- Training-set:\t\t{}".format(len(y_train)))
print("- Validation-set:\t{}".format(len(y_valid)))

# Hyper-parameters
learning_rate = 0.001   # The optimization initial learning rate
epochs = 10             # Total number of training epochs
batch_size = 100        # Training batch size
display_freq = 100      # Frequency of displaying the training results

# Network Parameters
input_dim = 28          # MNIST data input (img shape: 28*28)
max_time = 28           # Timesteps
num_hidden_units = 128  # Number of hidden units of the RNN
n_classes = 10          # Number of classes, one class per digit


# Create the graph for the linear model
# Placeholders for inputs (x) and outputs(y)
x = tf.placeholder(tf.float32, shape=[None, max_time, input_dim], name='X')
y = tf.placeholder(tf.float32, shape=[None, n_classes], name='Y')


# create weight matrix initialized randomely from N~(0, 0.01)
W = weight_variable(shape=[num_hidden_units, n_classes])

# create bias vector initialized as zero
b = bias_variable(shape=[n_classes])

output_logits = RNN(x, W, b, max_time, num_hidden_units)
y_pred = tf.nn.softmax(output_logits)

# Model predictions
cls_prediction = tf.argmax(output_logits, axis=1, name='predictions')

# Define the loss function, optimizer, and accuracy
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output_logits), name='loss')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(loss)
correct_prediction = tf.equal(tf.argmax(output_logits, 1), tf.argmax(y, 1), name='correct_pred')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

# Creating the op for initializing all variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    global_step = 0
    # Number of training iterations in each epoch
    num_tr_iter = int(len(y_train) / batch_size)
    for epoch in range(epochs):
        print('Training epoch: {}'.format(epoch + 1))
        x_train, y_train = randomize(x_train, y_train)
        for iteration in range(num_tr_iter):
            global_step += 1
            start = iteration * batch_size
            end = (iteration + 1) * batch_size
            x_batch, y_batch = get_next_batch(x_train, y_train, start, end)
            x_batch = x_batch.reshape((batch_size, max_time, input_dim))
            # Run optimization op (backprop)
            feed_dict_batch = {x: x_batch, y: y_batch}
            sess.run(optimizer, feed_dict=feed_dict_batch)

            if iteration % display_freq == 0:
                # Calculate and display the batch loss and accuracy
                loss_batch, acc_batch = sess.run([loss, accuracy],
                                                 feed_dict=feed_dict_batch)

                print("iter {0:3d}:\t Loss={1:.2f},\tTraining Accuracy={2:.01%}".
                      format(iteration, loss_batch, acc_batch))

        # Run validation after every epoch

        feed_dict_valid = {x: x_valid[:1000].reshape((-1, max_time, input_dim)), y: y_valid[:1000]}
        loss_valid, acc_valid = sess.run([loss, accuracy], feed_dict=feed_dict_valid)
        print('---------------------------------------------------------')
        print("Epoch: {0}, validation loss: {1:.2f}, validation accuracy: {2:.01%}".
              format(epoch + 1, loss_valid, acc_valid))
        print('---------------------------------------------------------')

    # Test the network after training
    # Accuracy
    x_test, y_test = load_data(mode='test')
    feed_dict_test = {x: x_test[:1000].reshape((-1, max_time, input_dim)), y: y_test[:1000]}
    loss_test, acc_test = sess.run([loss, accuracy], feed_dict=feed_dict_test)
    print('---------------------------------------------------------')
    print("Test loss: {0:.2f}, test accuracy: {1:.01%}".format(loss_test, acc_test))
    print('---------------------------------------------------------')
