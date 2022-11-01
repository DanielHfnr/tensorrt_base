

This repository contains a ROS catkin package as a library with a C++ base class for easier Tensorrt usage in other projects. Functionality includes parsing ONNX files and creating/saving/loading TensorRT engines from that as well as running inference in TensorRT.

**Tested with:**
- Ubuntu 20.04
- Nvidia Driver 520
- CUDA 11.6
- TensorRT 8.4.3
- ROS Noetic

### Prerequisites 

- Installed ROS: http://wiki.ros.org/noetic/Installation/Ubuntu
- Install CUDA: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
- Install TensorRT: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html

**Optional:**

Install clang-format for code formatting: 
```
sudo apt-get install clang-format
```

### Installation


Clone this repository into your catkin workspace:

```
git clone https://github.com/DanielHfnr/tensorrt_base.git
```

Build the package using `catkin_make` or `catkin build`. 


### Usage

TODO: Show image with implementation and usage of functions.

Provide link to other repo