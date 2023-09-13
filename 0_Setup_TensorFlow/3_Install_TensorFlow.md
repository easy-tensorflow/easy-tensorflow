# Install TensorFlow:

To install the library we will create a conda environment and name it __tensorflow__. However, you may choose your own desired name for it. In this example, we will install `tensorflow-2.10` for GPU; but you can select another version from. https://www.tensorflow.org/install/source#tested_build_configurations

Open command prompt (on Windows) or Terminal (on Linux) and type:
```bash
conda create --name tensorflow python=3.10
```

__*Note:__ make sure the version of python is compatible with the version of TensorFlow on the list. For example, for `tensorflow-2.10.0` we need __python 3.7__ to __python3.10__.

Once the environment is created, we can activate the environment:

`conda activate tensorflow`

At this step, the name of the environment will appear at the beginning of the line. such as:
```bash
(tensorflow) >>
```

Now you can go ahead and install the TensorFlow:

`pip install tensorflow==2.10`

__*Note:__ If you want to install the GPU version of TensorFlow, make sure you have installed CUDA and cuDNN as explained in the previous tutorial.