

# Install Python & Conda:
Conda package manager gives you the ability to __create multiple environments__ with different versions of Python and other libraries. This becomes useful when some codes are written with specific versions of a library. For example, you define your default TensorFlow environment with python 3.5 and TensorFlow 1.6  with GPU by the name ```tensorflow```. However, you may find another code that runs in python2.7 and has some functions that work with TensorFlow 1.2 with CPU. You can easily create a new environment and name it for example ```tf-12-cpu-py27```. In this way you don’t mess with your default environment and you can create multiple environments for multiple configurations. The figure below might give you some hints:

![Alt text](files/multi_env.png)


To install the Anaconda follow these steps:

1. Download the installer from [here](https://www.anaconda.com/download/).

2. Select the proper operating system.

    ![Alt text](files/conda_os.png)

3. Download the python 3.6 installer:

    ![Alt text](files/conda_ver.png)

4. Follow the instructions on installation in [here](https://docs.anaconda.com/anaconda/install/).

    __*Note:__ Remember the path that you are installing the Anaconda into. You will later need it for setting the path in PyCharm (we'll dive into it soon).

    __(For Windows):__ Make sure to select "Add Anaconda to my PATH environment variable".

    ![Alt text](files/conda_path.png)



__*Note:__ If you wanna learn more about Anaconda, watch [this](https://www.youtube.com/watch?v=YJC6ldI3hWk) amazing video which explains it thoroughly.
