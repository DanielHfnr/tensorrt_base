#include "tensorrt_base/TensorrtBase.h"

#include "NvInferPlugin.h"
#include "NvInferRuntimeCommon.h"
#include "NvOnnxParser.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h> // memcpy
#include <sys/stat.h>

std::string precisionTypeToStr(PrecisionType type)
{
    switch (type)
    {
    case TYPE_DISABLED: return "DISABLED";
    case TYPE_FASTEST: return "FASTEST";
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

TensorrtBase::TensorrtBase()
{
    engine_ = nullptr;
    infer_ = nullptr;
    context_ = nullptr;
    stream_ = nullptr;
    bindings_ = nullptr;

    precision_ = TYPE_FASTEST;
    device_ = DEVICE_GPU;
    allow_gpu_fallback_ = false;

    workspace_size_ = 32 << 24;
}

TensorrtBase::~TensorrtBase() {}

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
    char* engineStream = NULL;
    size_t engineSize = 0;

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
                onnx_model_path, precision, device, allow_gpu_fallback, calibrator, &engineStream, &engineSize))
        {
            gLogger.log(nvinfer1::ILogger::Severity::kERROR,
                ("Device %s failed to load model %s!", deviceTypeToStr(device), onnx_model_path).c_str());
            return false;
        }

        gLogger.log(nvinfer1::ILogger::Severity::kVERBOSE,
            ("Network profiling completed, saving engine to " + cache_engine_path_).c_str());

        // write the cache file
        FILE* cacheFile = NULL;
        cacheFile = fopen(cache_engine_path_.c_str(), "wb");

        if (cacheFile != NULL)
        {
            if (fwrite(engineStream, 1, engineSize, cacheFile) != engineSize)
            {
                gLogger.log(nvinfer1::ILogger::Severity::kERROR,
                    ("Failed to write engine cache file " + cache_engine_path_).c_str());
            }

            fclose(cacheFile);
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
        if (!LoadEngine(cache_engine_path_, &engineStream, &engineSize))
            return false;
    }

    gLogger.log(nvinfer1::ILogger::Severity::kVERBOSE, ("Device loaded model file " + onnx_model_path).c_str());

    if (!LoadEngine(engineStream, engineSize, NULL, device, cudaStreamDefault))
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR,
            ("Failed to create TensorRT engine on device " + deviceTypeToStr(device) + " from " + onnx_model_path)
                .c_str());
        return false;
    }

    gLogger.log(nvinfer1::ILogger::Severity::kVERBOSE, "Successfully initialized model...");

    free(engineStream); // not used anymore

    onnx_model_path_ = onnx_model_path;
    precision_ = precision;
    allow_gpu_fallback_ = allow_gpu_fallback;

    return true;
}

bool TensorrtBase::LoadEngine(char* engine_stream, size_t engine_size, nvinfer1::IPluginFactory* pluginFactory,
    DeviceType device, cudaStream_t stream)
{
    nvinfer1::IRuntime* infer = nvinfer1::createInferRuntime(gLogger);

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

    nvinfer1::ICudaEngine* engine = infer->deserializeCudaEngine(engine_stream, engine_size, pluginFactory);

    if (!engine)
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR,
            ("Failed to create CUDA engine on device " + deviceTypeToStr(device)).c_str());
        return false;
    }

    if (!engine)
        return NULL;

    nvinfer1::IExecutionContext* context = engine->createExecutionContext();

    if (!context)
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR,
            ("Failed to create execution context on device " + deviceTypeToStr(device)).c_str());
        return 0;
    }

    if (enable_debug_)
    {
        gLogger.log(nvinfer1::ILogger::Severity::kVERBOSE, "Enabling context debug sync...");
        context->setDebugSync(true);
    }

    // if( mEnableProfiler )
    //	context->setProfiler(&gProfiler);

    const int numBindings = engine->getNbBindings();

    for (int n = 0; n < numBindings; n++)
    {
        const int bind_index = n;
        const char* bind_name = engine->getBindingName(n);
        const nvinfer1::Dims bind_dims = engine->getBindingDimensions(n);
        const bool is_input = engine->bindingIsInput(n);

        gLogger.log(nvinfer1::ILogger::Severity::kVERBOSE,
            ("Binding Nr.: " + std::to_string(bind_index) + "    Name: " + std::string(bind_name)
                + "  Is input: " + std::to_string(is_input))
                .c_str());

        for (int i = 0; i < bind_dims.nbDims; i++)
        {
            gLogger.log(nvinfer1::ILogger::Severity::kVERBOSE,
                ("    -- dim " + std::to_string(i) + "   " + std::to_string(bind_dims.d[i])).c_str());
        }

        const size_t blobSize = sizeDims(bind_dims) * sizeof(float);

        // allocate output memory
        void* outputCPU = NULL;
        void* outputCUDA = NULL;

        if (!cudaAllocMapped((void**) &outputCPU, (void**) &outputCUDA, blobSize))
        {
            gLogger.log(nvinfer1::ILogger::Severity::kERROR,
                ("Failed to alloc CUDA mapped memory for tensor " + std::string(bind_name)).c_str());

            return false;
        }

        LayerInfo l;

        l.CPU = (float*) outputCPU;
        l.CUDA = (float*) outputCUDA;
        l.size = blobSize;
        l.name = bind_name;
        l.binding = bind_index;
        l.dims = bind_dims;

        if (is_input)
            inputs_.push_back(l);
        else
            outputs_.push_back(l);
    }

    const int bindingSize = numBindings * sizeof(void*);

    bindings_ = (void**) malloc(bindingSize);

    if (!bindings_)
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to allocate memory for bindings!");
        return false;
    }

    memset(bindings_, 0, bindingSize);

    for (uint32_t n = 0; n < GetNumInputLayers(); n++)
    {
        bindings_[inputs_[n].binding] = inputs_[n].CUDA;
    }

    for (uint32_t n = 0; n < GetNumOutputLayers(); n++)
    {
        bindings_[outputs_[n].binding] = outputs_[n].CUDA;
    }

    engine_ = engine;
    device_ = device;
    context_ = context;

    // SetStream(stream);	// set default device stream

    infer_ = infer;
    return true;
}

bool TensorrtBase::LoadEngine(std::string filename, char** stream, size_t* size)
{
    if (filename.empty() || !stream || !size)
        return false;

    char* engineStream = NULL;
    size_t engineSize = 0;

    // LogInfo(LOG_TRT "loading network plan from engine cache... %s\n", filename);
    gLogger.log(nvinfer1::ILogger::Severity::kVERBOSE, ("Loading model plan from engine cache " + filename).c_str());

    // determine the file size of the engine
    engineSize = fileSize(filename);

    if (engineSize == 0)
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR,
            ("Invalid engine cach file size " + std::to_string(engineSize)).c_str());
        return false;
    }

    // allocate memory to hold the engine
    engineStream = (char*) malloc(engineSize);

    if (!engineStream)
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR,
            ("Failed to allocate " + std::to_string(engineSize) + "bytes").c_str());
        return false;
    }

    // open the engine cache file from disk
    FILE* cacheFile = NULL;
    cacheFile = fopen(filename.c_str(), "rb");

    if (!cacheFile)
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, ("Failed to open engine cache file " + filename).c_str());
        return false;
    }

    // read the serialized engine into memory
    const size_t bytesRead = fread(engineStream, 1, engineSize, cacheFile);

    if (bytesRead != engineSize)
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR,
            ("Only read " + std::to_string(bytesRead) + "/" + std::to_string(engineSize)
                + " bytes of engine cache file " + filename)
                .c_str());
        return false;
    }

    // close the plan cache
    fclose(cacheFile);

    *stream = engineStream;
    *size = engineSize;

    return true;
}

bool TensorrtBase::ProfileModel(const std::string& onnx_model_file, // name for model
    PrecisionType precision, DeviceType device, bool allow_gpu_fallback, nvinfer1::IInt8Calibrator* calibrator,
    char** engine_stream, size_t* engine_size) // output stream for the GIE model
{
    if (!engine_stream || !engine_size)
        return false;

    // create builder and network definition interfaces
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);

    if (!builder)
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to create builder...");
        return false;
    }

    nvinfer1::INetworkDefinition* network
        = builder->createNetworkV2(1U << (uint32_t) nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    if (!network)
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Create network v2 failed...");
        return false;
    }

    gLogger.log(nvinfer1::ILogger::Severity::kVERBOSE, ("Parsing model file " + onnx_model_file).c_str());

    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);

    if (!parser)
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to create onnx parser...");
        return false;
    }

    const int parserLogLevel = (int) nvinfer1::ILogger::Severity::kVERBOSE;

    if (!parser->parseFromFile(onnx_model_file.c_str(), parserLogLevel))
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, ("Failed to parse model file " + onnx_model_file).c_str());
        return false;
    }

    // configure the builder
    nvinfer1::IBuilderConfig* builderConfig = builder->createBuilderConfig();

    builderConfig->setMaxWorkspaceSize(workspace_size_);

    builderConfig->setMinTimingIterations(3); // allow time for GPU to spin up
    builderConfig->setAvgTimingIterations(2);

    if (enable_debug_)
    {
        builderConfig->setFlag(nvinfer1::BuilderFlag::kDEBUG);
    }

    // set up the builder for the desired precision
    if (precision == TYPE_INT8)
    {
        builderConfig->setFlag(nvinfer1::BuilderFlag::kINT8);

        if (!calibrator)
        {
            gLogger.log(nvinfer1::ILogger::Severity::kERROR, "INT8 requested but INT8 calibrator is NULL...");
            return false;
        }

        builderConfig->setInt8Calibrator(calibrator);
    }
    else if (precision == TYPE_FP16)
    {
        builderConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    // set the default device type
    builderConfig->setDefaultDeviceType(deviceTypeToTRT(device));

    if (allow_gpu_fallback)
    {
        builderConfig->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
    }

    // build CUDA engine
    gLogger.log(nvinfer1::ILogger::Severity::kVERBOSE, "Building CUDA engine, this may take a while...");
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *builderConfig);

    if (!engine)
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR,
            ("Failed to build CUDA engine on device " + deviceTypeToStr(device)).c_str());
        return false;
    }

    gLogger.log(nvinfer1::ILogger::Severity::kVERBOSE,
        ("Done building CUDA engine on device " + deviceTypeToStr(device)).c_str());
    // we don't need the network definition any more, and we can destroy the parser
    network->destroy();

    // serialize the engine
    nvinfer1::IHostMemory* serMem = engine->serialize();

    if (!serMem)
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR,
            ("Failed to serialize CUDA engine on device " + deviceTypeToStr(device)).c_str());
        return false;
    }

    const char* serData = (char*) serMem->data();
    const size_t serSize = serMem->size();

    // allocate memory to store the bitstream
    char* engineMemory = (char*) malloc(serSize);

    if (!engineMemory)
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR,
            ("Failed to allocate " + std::to_string(serSize) + " bytes to store CUDA engine").c_str());
        return false;
    }

    memcpy(engineMemory, serData, serSize);

    *engine_stream = engineMemory;
    *engine_size = serSize;

    // free builder resources
    engine->destroy();
    builder->destroy();
    return true;
}

bool TensorrtBase::FileExists(const std::string& name)
{
    std::ifstream f(name.c_str());
    return f.good();
}

size_t TensorrtBase::fileSize(const std::string& path)
{
    if (path.size() == 0)
        return 0;

    struct stat fileStat;

    const int result = stat(path.c_str(), &fileStat);

    if (result == -1)
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, (path + " does not exist!").c_str());
        return 0;
    }

    return fileStat.st_size;
}

size_t TensorrtBase::sizeDims(const nvinfer1::Dims& dims, const size_t elementSize)
{
    size_t sz = dims.d[0];

    for (int n = 1; n < dims.nbDims; n++)
        sz *= dims.d[n];

    return sz * elementSize;
}

bool TensorrtBase::cudaAllocMapped(void** cpuPtr, void** gpuPtr, size_t size)
{
    if (!cpuPtr || !gpuPtr || size == 0)
        return false;

    if (cudaHostAlloc(cpuPtr, size, cudaHostAllocMapped) != cudaSuccess)
        return false;

    if (cudaHostGetDevicePointer(gpuPtr, *cpuPtr, 0) != cudaSuccess)
        return false;

    memset(*cpuPtr, 0, size);

    return true;
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
        if (!context_->execute(1, bindings_))
        {
            gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to execute TensorRT context!");
            return false;
        }
    }
    else
    {
        if (!context_->enqueue(1, bindings_, stream_, NULL))
        {
            gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to enqueue TensorRT context!");
            return false;
        }
    }

    return true;
}