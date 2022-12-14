#ifndef TENSORRTBASE_H
#define TENSORRTBASE_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>

// Check for non-NULL pointer before freeing it, and then set the pointer to NULL.
#define CUDA_FREE_HOST(x)                                                                                              \
    if (x != nullptr)                                                                                                  \
    {                                                                                                                  \
        cudaFreeHost(x);                                                                                               \
        x = nullptr;                                                                                                   \
    }

enum PrecisionType
{
    TYPE_DISABLED = 0, //!< Unknown, unspecified, or disabled type
    TYPE_FP32,         //!< 32-bit floating-point precision (FP32)
    TYPE_FP16,         //!< 16-bit floating-point half precision (FP16)
    TYPE_INT8,         //!< 8-bit integer precision (INT8)
    NUM_PRECISIONS     //!< Number of precision types defined
};

enum DeviceType
{
    DEVICE_GPU = 0, //!< GPU (if multiple GPUs are present, a specific GPU can be selected with cudaSetDevice()
    DEVICE_DLA,     //!< Deep Learning Accelerator (DLA) Core 0 (only on Jetson Xavier)
    DEVICE_DLA_0 = DEVICE_DLA, //!< Deep Learning Accelerator (DLA) Core 0 (only on Jetson Xavier)
    DEVICE_DLA_1,              //!< Deep Learning Accelerator (DLA) Core 1 (only on Jetson Xavier)
    NUM_DEVICES                //!< Number of device types defined
};

//!
//! \brief Function that returns a string from precisionType
//!
std::string precisionTypeToStr(PrecisionType type);

//!
//! \brief Function converts a string into precisionType
//!
PrecisionType precisionTypeFromStr(std::string str);

//!
//! \brief Function that returns a string from deviceType
//!
std::string deviceTypeToStr(DeviceType type);

//!
//! \brief Function that converts a string into deviceType
//!
DeviceType deviceTypeFromStr(std::string str);

//!
//! \brief TesnorRT logger class
//!
class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        // kINFO will not get printed
        if (severity == Severity::kINTERNAL_ERROR || severity == Severity::kERROR)
        {
            // Print in red
            std::cout << "\033[1;31m" + std::string(msg) + "\033[0m" << std::endl;
        }
        else if (severity == Severity::kWARNING)
        {
            // Print in yellow
            std::cout << "\033[1;33m" + std::string(msg) + "\033[0m" << std::endl;
        }
        else if (severity == Severity::kVERBOSE)
        {
            std::cout << msg << std::endl;
        }
    }
};

//!
//! \brief Abstract base class for loading ONNX networks and convert them into TensorRT engine and run inference.
//!
class TensorrtBase
{
public:
    TensorrtBase();
    ~TensorrtBase();

    //!
    //! \brief Load a new network instance of a onnx model
    //!
    //! \param onnx_model_path Filepath to the onnx model file
    //! \param precision Precision that should be used for engine created (FP32, FP16, INF8)
    //! \param device Device where the engine should be running on (GPU, DLA etc.)
    //! \param allow_gpu_fallback Boolean in GPU fallback should be allow for certain layers.
    //! \param calibrator Instance of a INT8 calibrator
    //!
    //! \return True if loading was successfull, Fals if not
    //!
    bool LoadNetwork(std::string onnx_model_path, PrecisionType precision, DeviceType device, bool allow_gpu_fallback,
        nvinfer1::IInt8Calibrator* calibrator = nullptr);

    //!
    //! \brief Runs inference on the created execution context
    //!
    //! \param stream With cuda stream given, inference should be asynchronously (non blocking execution). And needs to
    //! be synchronized afterwards. If not give, inference will be blocking until done.
    //!
    //! \return True if inference was successfull, False if not
    //!
    bool ProcessNetwork(cudaStream_t stream = nullptr);

protected:
    //!
    //! \brief Load a serialized engine plan file into memory.
    //!
    //! \param filename Filepath to the .engine file
    //! \param engine_blob Serialized Engine stream
    //!
    //! \return
    //!
    bool DeserializeEngine(std::string filename, std::vector<char>& engine_blob);

    //!
    //! \brief Load a network instance from a serialized engine plan file.
    //!
    //! \param engine_blob Deserialized Engine stream
    //! \param device Device type where engine should be create (GPU, DLA etc.)
    //!
    //! \return True if loading was successfull, False if not
    //!
    bool CreateInferenceEngine(std::vector<char>& engine_blob, DeviceType device);

    //!
    //! \brief Create and output an optimized network model
    //!
    //! \note this function is automatically used by LoadNetwork, but also can be used individually to perform the
    //! network operations offline.
    //!
    //! \param onnx_model_path Filepath to the onnx model file
    //! \param precision Precision that should be used for engine created (FP32, FP16, INF8)
    //! \param device Device where the engine should be running on (GPU, DLA etc.)
    //! \param allow_gpu_fallback Boolean in GPU fallback should be allow for certain layers.
    //! \param calibrator Instance of a INT8 calibrator
    //!
    //! \return True if profiling was successfull, False if not
    //!
    bool ProfileModel(const std::string& onnx_model_file, PrecisionType precision, DeviceType device,
        bool allow_gpu_fallback, nvinfer1::IInt8Calibrator* calibrator);

    //!
    //! \brief Checks if a file exists on filesystem
    //!
    //! \param path Filepath to the file
    //!
    //! \return True of file exists, False if not
    //!
    inline bool FileExists(const std::string& path);

    //!
    //! \brief Allocate ZeroCopy mapped memory, shared between CUDA and CPU.
    //!
    //! \note Although two pointers are returned, one for CPU and GPU, they both resolve to the same physical memory.
    //!
    //! \param cpuPtr Returned CPU pointer to the shared memory.
    //! \param gpuPtr Returned GPU pointer to the shared memory.
    //! \param size Size (in bytes) of the shared memory to allocate.
    //!
    //! \return `true` if the allocation succeeded, `false` otherwise.
    //!
    inline bool CudaAllocMapped(void** cpu_ptr, void** gpu_ptr, size_t size);

    //!
    //! \brief Copies memory from host to device
    //!
    //! \param dst Pointer to destination memory
    //! \param src pointer to source memory
    //! \param count Size in bytes
    //! \param stream Cuda stream. Copies asynchronously if stream is set. Otherwise blocking cudaMemCpy
    //!
    //! \return True if copy was successful, false if not
    //!
    bool CudaCopyToDevice(void* dst, const void* src, size_t count, cudaStream_t stream = nullptr);

    //!
    //! \brief Copies memory from device to host
    //!
    //! \param dst Pointer to destination memory
    //! \param src pointer to source memory
    //! \param count Size in bytes
    //! \param stream Cuda stream. Copies asynchronously if stream is set. Otherwise blocking cudaMemCpy
    //!
    //! \return True if copy was successful, false if not
    //!
    bool CudaCopyFromDevice(void* dst, const void* src, size_t count, cudaStream_t stream = nullptr);

    //!
    //! \brief Calculates the volume of the layer (all dimension sizes multiplied with each other)
    //!
    //! \param dims Dimensions of the layer
    //!
    //! \return Calculated volume of the layer
    //!
    inline size_t CalculateVolume(const nvinfer1::Dims& dims);

    //!
    //! \brief Returns the number of input layers in the model
    //!
    inline uint32_t GetNumInputLayers() const;

    //!
    //! \brief Returns the number of output layers in the model
    //!
    inline uint32_t GetNumOutputLayers() const;

    //!
    //! \brief Creates a new CUDA stream with or without arguments
    //!
    //! \param nonBlocking Flag if stream should be create with non blocking flag
    //!
    //! \return Created CUDA stream
    //!
    cudaStream_t CreateStream(bool nonBlocking);

    //!
    //! \brief Returns the dimension of a certain input layer
    //!
    //! \param input_layer_name Name of the input layer
    //!
    nvinfer1::Dims GetInputDims(std::string input_layer_name) const;

    //!
    //! \brief Returns the dimension of a certain output layer
    //!
    //! \param output_layer_name Name of the output layer
    //!
    nvinfer1::Dims GetOutputDims(std::string output_layer_name) const;

    //!
    //! \brief Serializes a loaded inference engine
    //!
    //! \param data Pointer to memory which should be serialized
    //! \param size Size of the data to be serialized
    //!
    bool SerializeEngine(void* data, size_t size);

protected:
    std::shared_ptr<nvinfer1::IExecutionContext> context_{nullptr}; //!< TensorRT execution context
    std::shared_ptr<nvinfer1::ICudaEngine> engine_{nullptr};        //!< TensorRT engine

    std::string onnx_model_path_;        //!< Filepath to the ONNX model given to the LoadNetwork function
    std::string cache_engine_path_;      //!< Constructed filepath from onnx model name and meta data like trt version
    PrecisionType precision_{TYPE_FP32}; //!< Precision for which the network will be built (FP32, FP16, INT8)
    DeviceType device_{DEVICE_GPU};      //!< Device where the trt engine will be executed
    bool allow_gpu_fallback_{false};     //!< If certain layers arent available e.g on DLA fallback to GPU
    bool enable_debug_{false};           //!< Boolean flag if debug sync is enabled on execution context
    std::vector<void*> bindings_{};      //!< Bindings of the trt engine (inputs, outputs)

    //!
    //! \brief
    //!
    struct LayerInfo
    {
        std::string name;    //!< Name of the layer as specified in the ONNX model
        nvinfer1::Dims dims; //!< Dimensions of the layer in NCHW
        uint32_t size;       //!< Size of the input/output blob in bytes
        uint32_t binding;    //!< Binding index
        void* CPU;           //!< CPU pointer to the cuda mapped memory
        void* CUDA;          //!< GPU pointer to the cuda mapped memory
    };

    std::map<std::string, LayerInfo> inputs_;  //!< Map of all input blobs
    std::map<std::string, LayerInfo> outputs_; //!< Map of all output blobs

    Logger gLogger; //!< TensorrRT logger class
};

#endif