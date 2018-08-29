# Install TensorFlow:

To install the library we will create an environment in Anaconda with __python 3.5__ we name it __tensorflow__. However, you may choose your own desired name for it. Open command prompt (or terminal) and type:
```bash
conda create --name tensorflow python=3.5
```
Once the environment is created, we can activate the environment:

__(for Windows):__
```bash
activate tensorflow
```
__(for Linux & Mac):__
```
source activate tensorflow
```
At this step, the name of the environment will appear at the beginning of the line. such as:
```bash
(tensorflow) >>
```
Now you can go ahead and install the TensorFlow:

__(for Windows):__

*(CPU version):*
```bash
pip install --upgrade tensorflow
```
*(GPU version):*
```bash
pip install --upgrade tensorflow-gpu
```
__(for Linux):__

*(CPU version):*
```bash
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.10.0-cp35-cp35m-linux_x86_64.whl
```
*(GPU version):*
```bash
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.10.0-cp35-cp35m-linux_x86_64.whl
 ```
__(for Mac):__

*(CPU version):*
```bash
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.10.0-py3-none-any.whl
```
