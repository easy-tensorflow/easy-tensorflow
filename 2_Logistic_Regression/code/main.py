import tensorflow as tf
import numpy as np
from utils import *
from ops import *

# Load MNIST data
x_train, y_train, x_valid, y_valid = load_data(mode='train')
print("Size of:")
print("- Training-set:\t\t{}".format(len(y_train)))
print("- Validation-set:\t{}".format(len(y_valid)))

# Data Dimensions
img_h = img_w = 28              # MNIST images are 28x28
img_size_flat = img_h * img_w   # 28x28=784, the total number of pixels
n_classes = 10                  # Number of classes, one class per digit

# Create the graph for the linear model
# Placeholders for inputs (x), outputs(y)
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='X')
y = tf.placeholder(tf.float32, shape=[None, n_classes], name='Y')




