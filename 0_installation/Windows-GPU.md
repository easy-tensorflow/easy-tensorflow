# Installing Tensorflow with GPU for Windows

## Requirements:

* Python 3.5
* CUDA-enabled GPU 
  * To verify you have the CUDA-enabled GPU open command prompt(click start and write `cmd` on search bar) and type:

	`control /name Microsoft.DeviceManager`

	You have to have a GPU that matches the list of CUDA-capable GPUs. You may find the list https://developer.nvidia.com/cuda-gpus.


## 1. Install CUDA Development Toolkit:

Tensorflow-gpu requires CUDA driver and cuDNN libraries. Tensorflow v1.4 is compatible with __CUDA toolkit 8.0__ and __cuDNN 6.0__. Tensorflow team anticipate releasing TensorFlow 1.5 with CUDA 9 and cuDNN 7.
 1. Download the CUDA toolkit 8.0 from  https://developer.nvidia.com/cuda-80-ga2-download-archive. Choose the correct version of your windows and select the local installer type.
 2. Download the cuDNN v6.0 (CUDA for Deep Neural Networks) library from https://developer.nvidia.com/cudnn
   * It will ask for setting up an account …
   * Download __cuDNN v6.0 for CUDA 8.0__
   * Copy the files to “C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0” in the corresponding folders

## 2. Install Anaconda:
The first thing you need to install the Tensorflow is to have the Anaconda distribution which handles the installation of python itself and the dependencies (libraries, etc.).

 1. Download the installer from https://www.anaconda.com/download
 2. Select the proper operating system

    ![Alt text](files/dl_os.png)

 3. Download the python 3.6 installer:

    ![Alt text](files/dl_ver.png)

 4. Follow the instructions on installation from https://docs.anaconda.com/anaconda/install/
    * Make sure to add Anaconda to my PATH environment variable.

    ![Alt text](files/dl_path.png)

## 3. Create Anaconda Environment:

One of the reason that we installed Anaconda is that (on Windows) we need python 3.5 to install the Tensorflow. So, go ahead and create an environment with python 3.5:
  1. Open command prompt 
  2. Create a new environment:

  `conda create -n tensorflow python=3.5`

  3. It will ask for installing new packages. Insert “y”.

      ![Alt text](files/conda_env_y.png)

## 4. Install Tensorflow:
Now that we have created the environment, we should activate it:

  `acativate tensorflow`

  * In this case your name of environment (which we named it tensorflow) will show in the beginning of the line.
  ![Alt text](files/cmd_change.png)


Now that the environment is activated we can go ahead and install the tensorflow-gpu:

`pip install tensorflow-gpu`

### Notes:
Always activate the environment before launching any editor.

`activate tensorflow`

There are some useful packages that we highly suggest to install:

`pip install matplotlib jupyter pillow`




	
