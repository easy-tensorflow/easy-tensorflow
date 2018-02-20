# Autoencoder

In __Neural Net__'s tutorial we saw that the network tries to predict the correct label corresponding to the input data. We saw that for MNIST dataset (which is a dataset of handwritten digits) we tried to predict the correct digit in the image. This type of machine learning algorithm is called __supervised learning__, simply because we are using __labels__.

__Autoencoder__ is neural networks that tries to __reconstruct the input data__. Since in training an __Autoencoder__ there are no labels involved, we have an __unsupervised learning__ method. __Autoencoders__ are mainly used for __Dimensionality reduction__. By encoding the input data to a new space (which we usually call ___latent space__) we will have a new representation of the data which is __compressed__ (so it uses less memory), __more abstract__ (contains more essential information of data). 

__Autoencoder__ help us dealing with noisy data. Since the __latent space__ only keeps the important information, the noise will not be preserved in the space and we can reconstruct the cleaned data.

__Autoencoder__ also helps us to understand how the neural networks work. We can visualize what a node has been experted on. This will give us an intuitive about the way these networks perform.


In this tutorial we will implement:
1. Autoencoder for noise removal (AE_noiseRemoval.ipynb)
2. Visualization of a node's activation (AE_visActivation)

## References:
* [www.tensorflow.com](www.tensorflow.com)
* [Visualizing a Trained Autoencoder](http://ufldl.stanford.edu/wiki/index.php/Visualizing_a_Trained_Autoencoder)