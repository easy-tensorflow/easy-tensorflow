import tensorflow as tf
import numpy as np
from ops import fc_layer
from utils import *

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print("Size of:")
print("- Training-set:\t\t{}".format(len(mnist.train.labels)))
print("- Test-set:\t\t{}".format(len(mnist.test.labels)))
print("- Validation-set:\t{}".format(len(mnist.validation.labels)))

# Data Dimensions
img_h = img_w = 28              # MNIST images are 28x28
img_size_flat = img_h * img_w   # 28x28=784, the total number of pixels
n_classes = 10                  # Number of classes, one class per digit

# Network Configuration
# 1st Convolutional Layer
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.
stride1 = 1               # The stride of the sliding window

# 2nd Convolutional Layer
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 32         # There are 32 of these filters.
stride2 = 1               # The stride of the sliding window

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.

# Create the network graph
# Placeholders for inputs (x), outputs(y)
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='X')
y = tf.placeholder(tf.float32, shape=[None, n_classes], name='Y')