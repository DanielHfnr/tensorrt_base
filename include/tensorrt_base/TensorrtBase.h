#ifndef TENSORRTBASE_H
#define TENSORRTBASE_H

#include <fstream>
#include <iostream>
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

class TensorrtBase
{
public:
    TensorrtBase();
    ~TensorrtBase();

    //!
    //! \brief
    //!
    //! \param onnx_model_path
    //! \param precision
    //! \param device
    //! \param allow_gpu_fallback
    //! \param calibrator
    //!
    //! \return
    //!
    bool LoadNetwork(std::string onnx_model_path, PrecisionType precision, DeviceType device, bool allow_gpu_fallback,
        nvinfer1::IInt8Calibrator* calibrator = nullptr);

    //!
    //! \brief
    //!
    //! \param sync
    //!
    //! \return
    //!
    bool ProcessNetwork(bool sync = true);

protected:
    //!
    //! \brief
    //!
    //! \param filename
    //! \param stream
    //! \param size
    //!
    //! \return
    //!
    bool LoadEngine(std::string filename, char** stream, size_t* size);

    //!
    //! \brief
    //!
    //! \param engine_stream
    //! \param engine_size
    //! \param plugin_factory
    //! \param device
    //!
    //! \return
    //!
    bool LoadEngine(char* engine_stream, size_t engine_size, DeviceType device);

    //!
    //! \brief
    //!
    //! \param onnx_model_file
    //! \param precision
    //! \param device
    //! \param allow_gpu_fallback
    //! \param calibrator
    //! \param engine_stream
    //! \param engine_size
    //!
    //! \return
    //!
    bool ProfileModel(const std::string& onnx_model_file, PrecisionType precision, DeviceType device,
        bool allow_gpu_fallback, nvinfer1::IInt8Calibrator* calibrator, char** engine_stream, size_t* engine_size);

    //!
    //! \brief
    //!
    //! \param name
    //!
    //! \return
    //!
    inline bool FileExists(const std::string& name);

    //!
    //! \brief
    //!
    //! \param path
    //!
    //! \return
    //!
    size_t FileSize(const std::string& path);

    //!
    //! \brief
    //!
    //! \param cpuPtr
    //! \param gpuPtr
    //! \param size
    //!
    //! \return
    //!
    inline bool CudaAllocMapped(void** cpu_ptr, void** gpu_ptr, size_t size);

    //!
    //! \brief
    //!
    //! \param dims
    //! \param element_size
    //!
    //! \return
    //!
    inline size_t SizeDims(const nvinfer1::Dims& dims, const size_t element_size = 1);

    //!
    //! \brief
    //!
    //! \return
    //!
    inline uint32_t GetNumInputLayers() const;

    //!
    //! \brief
    //!
    //! \return
    //!
    inline uint32_t GetNumOutputLayers() const;

    //!
    //! \brief
    //!
    //! \param nonBlocking
    //!
    //! \return
    //!
    cudaStream_t CreateStream(bool nonBlocking);

    //!
    //! \brief
    //!
    //! \param stream
    //!
    void SetStream(const cudaStream_t stream);

    //!
    //! \brief
    //!
    //! \return
    //!
    cudaStream_t GetStream() const;

protected:
    std::shared_ptr<nvinfer1::IExecutionContext> context_{nullptr}; //!< TensorRT execution context
    std::shared_ptr<nvinfer1::ICudaEngine> engine_{nullptr};        //!< TensorRT engine

    std::string onnx_model_path_;        //!< Filepath to the ONNX model given to the LoadNetwork function
    std::string cache_engine_path_;      //!< Constructed filepath from onnx model name and meta data like trt version
    PrecisionType precision_{TYPE_FP32}; //!< Precision for which the network will be built (FP32, FP16, INT8)
    DeviceType device_{DEVICE_GPU};      //!< Device where the trt engine will be executed
    bool allow_gpu_fallback_{false};     //!< If certain layers arent available e.g on DLA fallback to GPU
    bool enable_debug_{false};           //!< Boolean flag if debug sync is enabled on execution context
    void** bindings_{nullptr};           //!< Bindings of the trt engine (inputs, outputs)
    cudaStream_t stream_{nullptr};       //!< Current CUDA stream

    //!
    //! \brief
    //!
    struct LayerInfo
    {
        std::string name;    //!< Name of the layer as specified in the ONNX model
        nvinfer1::Dims dims; //!< Dimensions of the layer in NCHW
        uint32_t size;       //!< Size of the input/output blob in bytes
        uint32_t binding;    //!< Binding index
        float* CPU;          //!< CPU pointer to the cuda mapped memory
        float* CUDA;         //!< GPU pointer to the cuda mapped memory
    };

    std::vector<LayerInfo> inputs_;  //!< Vector of all input blobs
    std::vector<LayerInfo> outputs_; //!< Vector of all output blobs

    Logger gLogger; //!< TensorrRT logger class
};

#endif