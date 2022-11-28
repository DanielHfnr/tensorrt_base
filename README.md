

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

Follow the instructions to install ROS, CUDA and TensorRT.

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

Implement a C++ class and inherit from the TensorrtBase class like this: 


```
class TensorrtYolo final : public TensorrtBase
```

To initialize the model you can simply call the `LoadNetwork` function like this: 


```
// General model loading
if (!LoadNetwork(onnx_model_path, precision, device, allow_gpu_fallback))
{
    gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to load network...");
    return false;
}
```

Before running inference you must preprocess you inputs and copy your input data to the respective buffers. You can access pointers to input buffers really easily like this:

```
inputs_["input_layer_name_in_onnx_model"].CPU
```

After you copied your input data run the inference using this command:

```
if (!ProcessNetwork())
{
    gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to run inference...");
    return false;
}
```

After inference you can access the networks results in the same way as you got to the input:

```
outputs_["output_layer_name_in_onnx_model"].CPU
```

An example implementation for YoloV7 can be found here: [https://github.com/DanielHfnr/tensorrt_yolo_ros](https://github.com/DanielHfnr/tensorrt_yolo_ros)

