#include "tensorrt_base/TensorrtBase.h"

#include "NvInferPlugin.h"
#include "NvInferRuntimeCommon.h"
#include "NvOnnxParser.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <memory>
#include <string.h> // memcpy

std::string precisionTypeToStr(PrecisionType type)
{
    switch (type)
    {
    case TYPE_DISABLED: return "DISABLED";
    case TYPE_FP32: return "FP32";
    case TYPE_FP16: return "FP16";
    case TYPE_INT8: return "INT8";
    }
    return "";
}

PrecisionType precisionTypeFromStr(std::string str)
{
    if (str.empty())
        return TYPE_DISABLED;

    for (int n = 0; n < NUM_PRECISIONS; n++)
    {
        if (str == precisionTypeToStr((PrecisionType) n))
            return (PrecisionType) n;
    }

    return TYPE_DISABLED;
}

std::string deviceTypeToStr(DeviceType type)
{
    switch (type)
    {
    case DEVICE_GPU: return "GPU";
    case DEVICE_DLA_0: return "DLA_0";
    case DEVICE_DLA_1: return "DLA_1";
    }
    return "";
}

DeviceType deviceTypeFromStr(std::string str)
{
    if (str.empty())
        return DEVICE_GPU;

    for (int n = 0; n < NUM_DEVICES; n++)
    {
        if (str == deviceTypeToStr((DeviceType) n))
            return (DeviceType) n;
    }

    if (str == "DLA")
        return DEVICE_DLA;

    return DEVICE_GPU;
}

nvinfer1::DeviceType deviceTypeToTRT(DeviceType type)
{
    switch (type)
    {
    case DEVICE_GPU: return nvinfer1::DeviceType::kGPU;
    case DEVICE_DLA_0: return nvinfer1::DeviceType::kDLA;
    case DEVICE_DLA_1: return nvinfer1::DeviceType::kDLA;
    default: return nvinfer1::DeviceType::kGPU;
    }
}

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        delete obj;
    }
};

TensorrtBase::TensorrtBase() {}

TensorrtBase::~TensorrtBase()
{
    for (const auto& kv : inputs_)
    {
        LayerInfo layer_info = kv.second;
        CUDA_FREE_HOST(layer_info.CUDA);
    }

    for (const auto& kv : outputs_)
    {
        LayerInfo layer_info = kv.second;
        CUDA_FREE_HOST(layer_info.CUDA);
    }

    free(bindings_);
}

bool TensorrtBase::LoadNetwork(std::string onnx_model_path, PrecisionType precision, DeviceType device,
    bool allow_gpu_fallback, nvinfer1::IInt8Calibrator* calibrator)
{
    // TODO: Implement checks if everythin is filled correctly

    // Check if plugins are loaded correctly
    gLogger.log(nvinfer1::ILogger::Severity::kVERBOSE, "Loading NVIDIA plugins...");
    bool plugins_loaded = initLibNvInferPlugins(&gLogger, "");

    if (!plugins_loaded)
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to load NVIDIA plugins.");
    else
        gLogger.log(nvinfer1::ILogger::Severity::kVERBOSE, "Completed loading NVIDIA plugins.");

    // Try to load engine
    char* engine_stream = NULL;
    size_t engine_size = 0;

    std::string cache_prefix = onnx_model_path + "." + std::to_string((uint32_t) allow_gpu_fallback) + "."
        + std::to_string(NV_TENSORRT_VERSION) + "." + deviceTypeToStr(device) + "." + precisionTypeToStr(precision);

    cache_engine_path_ = cache_prefix + ".engine";

    // check for existence of cache
    if (!FileExists(cache_engine_path_))
    {
        gLogger.log(nvinfer1::ILogger::Severity::kWARNING,
            ("Cache file invalid, profiling model on device " + deviceTypeToStr(device)).c_str());

        // check for existence of model
        if (onnx_model_path.empty())
        {
            gLogger.log(nvinfer1::ILogger::Severity::kERROR, ("ONNX model file not found: " + onnx_model_path).c_str());
            return false;
        }

        // parse the model and profile the engine
        if (!ProfileModel(
                onnx_model_path, precision, device, allow_gpu_fallback, calibrator, &engine_stream, &engine_size))
        {
            gLogger.log(nvinfer1::ILogger::Severity::kERROR,
                ("Device %s failed to load model %s!", deviceTypeToStr(device), onnx_model_path).c_str());
            return false;
        }

        gLogger.log(nvinfer1::ILogger::Severity::kVERBOSE,
            ("Network profiling completed, saving engine to " + cache_engine_path_).c_str());

        // write the cache file
        FILE* cache_file = NULL;
        cache_file = fopen(cache_engine_path_.c_str(), "wb");

        if (cache_file != NULL)
        {
            if (fwrite(engine_stream, 1, engine_size, cache_file) != engine_size)
            {
                gLogger.log(nvinfer1::ILogger::Severity::kERROR,
                    ("Failed to write engine cache file " + cache_engine_path_).c_str());
            }

            fclose(cache_file);
        }
        else
        {
            gLogger.log(nvinfer1::ILogger::Severity::kERROR,
                ("Failed to open engine cache file for writing: " + cache_engine_path_).c_str());
        }

        gLogger.log(nvinfer1::ILogger::Severity::kVERBOSE, "Completed saving engine...");
    }
    else
    {
        if (!LoadEngine(cache_engine_path_, &engine_stream, &engine_size))
            return false;
    }

    gLogger.log(nvinfer1::ILogger::Severity::kVERBOSE, ("Device loaded model file " + onnx_model_path).c_str());

    if (!LoadEngine(engine_stream, engine_size, device))
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR,
            ("Failed to create TensorRT engine on device " + deviceTypeToStr(device) + " from " + onnx_model_path)
                .c_str());
        return false;
    }

    gLogger.log(nvinfer1::ILogger::Severity::kVERBOSE, "Successfully initialized model...");

    free(engine_stream); // not used anymore

    onnx_model_path_ = onnx_model_path;
    precision_ = precision;
    allow_gpu_fallback_ = allow_gpu_fallback;
    device_ = device;

    return true;
}

bool TensorrtBase::LoadEngine(char* engine_stream, size_t engine_size, DeviceType device)
{
    auto infer = std::unique_ptr<nvinfer1::IRuntime, InferDeleter>(nvinfer1::createInferRuntime(gLogger));

    if (!infer)
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR,
            ("Failed to create TRT engine on device " + deviceTypeToStr(device)).c_str());
        return false;
    }

    // if using DLA, set the desired core before deserialization occurs
    if (device == DEVICE_DLA_0)
    {
        gLogger.log(nvinfer1::ILogger::Severity::kVERBOSE, "Enabling DLA core 0...");
        infer->setDLACore(0);
    }
    else if (device == DEVICE_DLA_1)
    {
        gLogger.log(nvinfer1::ILogger::Severity::kVERBOSE, "Enabling DLA core 1...");
        infer->setDLACore(1);
    }

    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
        infer->deserializeCudaEngine(engine_stream, engine_size), InferDeleter());

    if (!engine_)
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR,
            ("Failed to create CUDA engine on device " + deviceTypeToStr(device)).c_str());
        return false;
    }

    context_ = std::shared_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext(), InferDeleter());

    if (!context_)
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR,
            ("Failed to create execution context on device " + deviceTypeToStr(device)).c_str());
        return 0;
    }

    if (enable_debug_)
    {
        gLogger.log(nvinfer1::ILogger::Severity::kVERBOSE, "Enabling context debug sync...");
        context_->setDebugSync(true);
    }

    // if( mEnableProfiler )
    //	context->setProfiler(&gProfiler);

    const int num_bindings = engine_->getNbBindings();

    for (int n = 0; n < num_bindings; n++)
    {
        const int bind_index = n;
        const char* bind_name = engine_->getBindingName(n);
        const nvinfer1::Dims bind_dims = engine_->getBindingDimensions(n);
        const bool is_input = engine_->bindingIsInput(n);

        gLogger.log(nvinfer1::ILogger::Severity::kVERBOSE,
            ("Binding Nr.: " + std::to_string(bind_index) + "    Name: " + std::string(bind_name)
                + "  Is input: " + std::to_string(is_input))
                .c_str());

        for (int i = 0; i < bind_dims.nbDims; i++)
        {
            gLogger.log(nvinfer1::ILogger::Severity::kVERBOSE,
                ("    -- dim " + std::to_string(i) + "   " + std::to_string(bind_dims.d[i])).c_str());
        }

        const size_t blob_size = CalculateVolume(bind_dims) * sizeof(float);

        gLogger.log(nvinfer1::ILogger::Severity::kVERBOSE,
            ("Alloc CUDA mapped memory for tensor with size (bytes): " + std::to_string(blob_size)).c_str());

        // allocate output memory
        void* output_cpu = NULL;
        void* output_cuda = NULL;

        if (!CudaAllocMapped((void**) &output_cpu, (void**) &output_cuda, blob_size))
        {
            gLogger.log(nvinfer1::ILogger::Severity::kERROR,
                ("Failed to alloc CUDA mapped memory for tensor " + std::string(bind_name)).c_str());

            return false;
        }

        LayerInfo l;

        l.CPU = (float*) output_cpu;
        l.CUDA = (float*) output_cuda;
        l.size = blob_size;
        l.name = bind_name;
        l.binding = bind_index;
        l.dims = bind_dims;

        if (is_input)
        {
            inputs_.emplace(bind_name, l);
        }
        else
        {
            outputs_.emplace(bind_name, l);
        }
    }

    const int binding_size = num_bindings * sizeof(void*);

    bindings_ = (void**) malloc(binding_size);

    if (!bindings_)
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to allocate memory for bindings!");
        return false;
    }

    memset(bindings_, 0, binding_size);

    for (const auto& kv : inputs_)
    {
        LayerInfo layer_info = kv.second;
        bindings_[layer_info.binding] = layer_info.CUDA;
    }

    for (const auto& kv : outputs_)
    {
        LayerInfo layer_info = kv.second;
        bindings_[layer_info.binding] = layer_info.CUDA;
    }

    return true;
}

bool TensorrtBase::LoadEngine(std::string filename, char** stream, size_t* size)
{
    if (filename.empty() || !stream || !size)
        return false;

    char* engine_stream = NULL;
    size_t engine_size = 0;

    gLogger.log(nvinfer1::ILogger::Severity::kVERBOSE, ("Loading model plan from engine cache " + filename).c_str());

    std::ifstream engine_file(filename, std::ios::binary);

    if (!engine_file.good())
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, ("Failed to open engine file: " + filename).c_str());
    }

    // Check the size of the file
    engine_file.seekg(0, std::ifstream::end);
    engine_size = engine_file.tellg();
    engine_file.seekg(0, std::ifstream::beg);

    if (engine_size == 0)
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR,
            ("Invalid engine cach file size " + std::to_string(engine_size)).c_str());
        return false;
    }

    // allocate memory to hold the engine
    engine_stream = (char*) malloc(engine_size);

    if (!engine_stream)
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR,
            ("Failed to allocate " + std::to_string(engine_size) + "bytes").c_str());
        return false;
    }

    engine_file.read(reinterpret_cast<char*>(engine_stream), engine_size);

    if (!engine_file.good())
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to read engine cache file");
    }

    engine_file.close();

    *stream = engine_stream;
    *size = engine_size;

    return true;
}

bool TensorrtBase::ProfileModel(const std::string& onnx_model_file, // name for model
    PrecisionType precision, DeviceType device, bool allow_gpu_fallback, nvinfer1::IInt8Calibrator* calibrator,
    char** engine_stream, size_t* engine_size) // output stream for the GIE model
{
    if (!engine_stream || !engine_size)
        return false;

    // create builder and network definition interfaces
    auto builder = std::unique_ptr<nvinfer1::IBuilder, InferDeleter>(nvinfer1::createInferBuilder(gLogger));

    if (!builder)
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to create builder...");
        return false;
    }

    auto network = std::unique_ptr<nvinfer1::INetworkDefinition, InferDeleter>(
        builder->createNetworkV2(1U << (uint32_t) nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

    if (!network)
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Create network v2 failed...");
        return false;
    }

    gLogger.log(nvinfer1::ILogger::Severity::kVERBOSE, ("Parsing model file " + onnx_model_file).c_str());

    auto parser = std::unique_ptr<nvonnxparser::IParser, InferDeleter>(nvonnxparser::createParser(*network, gLogger));

    if (!parser)
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to create onnx parser...");
        return false;
    }

    const int parser_log_level = (int) nvinfer1::ILogger::Severity::kVERBOSE;

    if (!parser->parseFromFile(onnx_model_file.c_str(), parser_log_level))
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, ("Failed to parse model file " + onnx_model_file).c_str());
        return false;
    }

    // configure the builder
    auto builder_config = std::unique_ptr<nvinfer1::IBuilderConfig, InferDeleter>(builder->createBuilderConfig());

    builder_config->setAvgTimingIterations(2);

    if (enable_debug_)
    {
        builder_config->setFlag(nvinfer1::BuilderFlag::kDEBUG);
    }

    // set up the builder for the desired precision
    if (precision == TYPE_INT8)
    {
        builder_config->setFlag(nvinfer1::BuilderFlag::kINT8);

        if (!calibrator)
        {
            gLogger.log(nvinfer1::ILogger::Severity::kERROR, "INT8 requested but INT8 calibrator is NULL...");
            return false;
        }

        builder_config->setInt8Calibrator(calibrator);
    }
    else if (precision == TYPE_FP16)
    {
        builder_config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    // set the default device type
    builder_config->setDefaultDeviceType(deviceTypeToTRT(device));

    if (allow_gpu_fallback)
    {
        builder_config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
    }

    // build CUDA engine
    gLogger.log(nvinfer1::ILogger::Severity::kVERBOSE, "Building serialized network, this may take a while...");

    auto serialize_memory = std::unique_ptr<nvinfer1::IHostMemory, InferDeleter>(
        builder->buildSerializedNetwork(*network, *builder_config));

    if (!serialize_memory)
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR,
            ("Failed to build network on device " + deviceTypeToStr(device)).c_str());
        return false;
    }

    const char* serialize_data = (char*) serialize_memory->data();
    const size_t serialize_size = serialize_memory->size();

    // allocate memory to store the bitstream
    char* engine_memory = (char*) malloc(serialize_size);

    if (!engine_memory)
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR,
            ("Failed to allocate " + std::to_string(serialize_size) + " bytes to store CUDA engine").c_str());
        return false;
    }

    memcpy(engine_memory, serialize_data, serialize_size);

    *engine_stream = engine_memory;
    *engine_size = serialize_size;

    return true;
}

bool TensorrtBase::FileExists(const std::string& path)
{
    std::ifstream f(path.c_str());
    return f.good();
}

size_t TensorrtBase::CalculateVolume(const nvinfer1::Dims& dims)
{
    size_t sz = dims.d[0];

    for (int n = 1; n < dims.nbDims; n++)
        sz *= dims.d[n];

    return sz;
}

bool TensorrtBase::CudaAllocMapped(void** cpu_ptr, void** gpu_ptr, size_t size)
{
    if (!cpu_ptr || !gpu_ptr || size == 0)
        return false;

    if (cudaHostAlloc(cpu_ptr, size, cudaHostAllocMapped) != cudaSuccess)
        return false;

    if (cudaHostGetDevicePointer(gpu_ptr, *cpu_ptr, 0) != cudaSuccess)
        return false;

    memset(*cpu_ptr, 0, size);

    return true;
}

cudaStream_t TensorrtBase::CreateStream(bool nonBlocking)
{
    uint32_t flags = cudaStreamDefault;

    if (nonBlocking)
        flags = cudaStreamNonBlocking;

    cudaStream_t stream = nullptr;

    if (cudaStreamCreateWithFlags(&stream, flags) != cudaSuccess)
        return nullptr;

    return stream;
}

void TensorrtBase::SetStream(const cudaStream_t stream)
{
    stream_ = stream;
}

cudaStream_t TensorrtBase::GetStream() const
{
    return stream_;
}

nvinfer1::Dims TensorrtBase::GetInputDims(std::string input_layer_name) const
{
    if (!input_layer_name.empty())
    {
        // Ensure the key is in the map before accessing it
        if (inputs_.count(input_layer_name))
        {
            return inputs_.find(input_layer_name)->second.dims;
        }
    }

    return nvinfer1::Dims{};
}

nvinfer1::Dims TensorrtBase::GetOutputDims(std::string output_layer_name) const
{
    if (!output_layer_name.empty())
    {
        // Ensure the key is in the map before accessing it
        if (outputs_.count(output_layer_name))
        {
            return outputs_.find(output_layer_name)->second.dims;
        }
    }

    return nvinfer1::Dims{};
}

uint32_t TensorrtBase::GetNumInputLayers() const
{
    return inputs_.size();
}

uint32_t TensorrtBase::GetNumOutputLayers() const
{
    return outputs_.size();
}

bool TensorrtBase::ProcessNetwork(bool sync)
{
    if (sync)
    {
        if (!context_->executeV2(bindings_))
        {
            gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to execute TensorRT context!");
            return false;
        }
    }
    else
    {
        if (!context_->enqueueV2(bindings_, stream_, NULL))
        {
            gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to enqueue TensorRT context!");
            return false;
        }
    }

    return true;
}