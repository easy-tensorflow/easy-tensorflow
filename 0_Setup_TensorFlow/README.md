
# How to setup TensorFlow on your Machine
TIn this set of tutorials, we explain how to setup your machine to run __TensorFlow__ codes "step by step".

### What is TensorFlow?
TensorFlow is a machine learning / deep learning library developed by Google that ...

[![ALT TEXT](http://img.youtube.com/vi/oZikw5k_2FM/0.jpg)](http://www.youtube.com/watch?v=oZikw5k_2FM "What is TensorFlow")

### Why TensorFlow?
Deep learning has found it's way to different branches of science. Well, let's see some applications of TensorFlow...
[![ALT TEXT](http://img.youtube.com/vi/mWl45NkFBOc/0.jpg)](http://www.youtube.com/watch?v=mWl45NkFBOc "Why TensorFlow")

Similar to many other libraries, we tried installing many side packages and libraries and experienced lots of problems and errors. We finally came up with a general solution and recommend installing the following libraries and packages as the __best way__ around it.

## 0. CUDA-toolkit & cuDNN library (for GPU version of TensorFlow only)

TensorFlow comes with two versions.
1. __CPU version__: Is easy to install but it is slow.

2. __GPU version__: Is tricky to install but it is fast.

To use the GPU version, you should make sure your machine has a cuda enabled GPU and both CUDA-tooklit and cuDNN are installed on your machine properly.

__Follow [this](0_CUDA_cuDNN.md) instruction to install the CUDA-toolkit and cuDNN library.__

## 1. Python
#### 1.1. Python programing language
TensorFlow has several APIs (Application Program Interface). But python API is the most complete and easiest to use <sup id="a1">[1](#f1)</sup>
. Python comes pre-installed with most Linux and Mac distributions. However, here we will install the python via __Anaconda__ distribution because it gives the flexibility to create __multiple environments__ for different versions of python and libraries.

__TensorFlow used to run only with python 3.5 on windows__. But recently they added the support for both 3.5 and 3.6. We will use Python 3.5 for all operating systems (Windows, Linux, and Mac) to keep it uniform among OSs throughout the tutorial. But feel free to use your own preferred python version.

If you are interested to learn more about python basics, we suggest you these tutorials:

- [sendtex](https://pythonprogramming.net/python-fundamental-tutorials/)
- [corey schafer](https://www.youtube.com/watch?v=YYXdXT2l-Gg&list=PL-osiE80TeTt2d9bfVyTiXJA-UTHn6WwU)

#### 1.2. Package manager
To run TensorFlow, you need to install the library. Libraries are also called packages. So, you need to have a __package management system__. There are 2 famous package management system:
1. __Pip:__ is the default package management system that comes with python. Pip installs python packages only and builds from the source. So, if you want to install a package, you have to make sure you have all the dependencies. For example, if you want to install _tflearn_ package, you have to make sure you have already installed _tensorflow_. Otherwise, you will get errors running _tflearn_ codes.

2. __Conda:__ is the package manager from Anaconda distribution. conda can be used for any software. Conda installs binaries meaning that it skips the compilation of the source code. If you don't want to deal with dependencies, it is better to install your package with conda. For example, if you want to install _tflearn_ package, you do not need to worry about installing _tensorflow_ package. It will automatically install all the needed packages. But, if you have a GPU in your systam and the binary file is build based on CPU version of the _tensorflow_ you will not be able to use the GPU version. Otherwise, you have to find the proper binary which has been built on GPU version.

__Follow [this](1_Install_Python_Anaconda.md) instruction to install python and conda.__

## 2. TensorFlow library
Now, having installed all the prerequisites, you can start installing the __TensorFlow__ library.

__Follow [this](2_Install_TensorFlow.md) instruction to install TensorFlow.__

## 3. Integrated Development Environment (IDE)
Now that the TensorFlow is installed in your machine. You can start coding. You can write your codes in any editor (terminal, emacs, notepad, ...). We suggest using __PyCharm__ because it offers a powerful debugging tool which is very useful especially when you write codes in TensorFlow.

__Follow [this](3_Install_PyCharm.md) instruction to install PyCharm.__

## 4. Run a sample code
Write a short program like the following and run it to check everything is working fine:
```python
import tensorflow as tf
a = tf.constant(2)
with tf.Session() as sess:
   print(sess.run(a))
```
```python
2
```
It must print out the value of a, 2.

### Final note
We suggest you to install some useful packages throughout these tutorials. In your terminal, activate the ```tensorflow``` environment and install the following packages:

__(for Windows):__
```bash
activate tensorflow
pip install matplotlib jupyter
```

__(for Linux & Mac):__
```bash
source activate tensorflow
pip install matplotlib jupyter
```

## References:

<b id="f1">[1]</b>: https://www.tensorflow.org/api_docs/ [↩](#a1)


Thanks for reading! If you have any question or doubt, feel free to leave a comment in our [website](http://easy-tensorflow.com/).
