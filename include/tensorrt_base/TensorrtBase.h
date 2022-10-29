#ifndef TENSORRTBASE_H
#define TENSORRTBASE_H

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <NvInfer.h>

enum PrecisionType
{
    TYPE_DISABLED = 0, //!< Unknown, unspecified, or disabled type
    TYPE_FASTEST,      //!< The fastest detected precision should be use (i.e. try INT8, then FP16, then FP32)
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
        if (severity != Severity::kINFO)
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
    //! \param pluginFactory
    //! \param device
    //! \param stream
    //!
    //! \return
    //!
    bool LoadEngine(char* engine_stream, size_t engine_size, nvinfer1::IPluginFactory* pluginFactory, DeviceType device,
        cudaStream_t stream);

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
    size_t fileSize(const std::string& path);

    //!
    //! \brief
    //!
    //! \param cpuPtr
    //! \param gpuPtr
    //! \param size
    //!
    //! \return
    //!
    inline bool cudaAllocMapped(void** cpuPtr, void** gpuPtr, size_t size);

protected:
    nvinfer1::IRuntime* infer_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;

    std::string onnx_model_path_;
    std::string cache_engine_path_;
    std::string checksum_path_;
    PrecisionType precision_;
    DeviceType device_;
    bool allow_gpu_fallback_;
    bool enable_debug_;
    void** bindings_;

    cudaStream_t stream_;
    uint32_t workspace_size_;

    struct LayerInfo
    {
        std::string name;
        nvinfer1::Dims dims;
        uint32_t size;
        uint32_t binding;
        float* CPU;
        float* CUDA;
    };

    std::vector<LayerInfo> inputs_;  //!< Vector of all input blobs
    std::vector<LayerInfo> outputs_; //!< Vector of all output blobs

    Logger gLogger;
};

#endif