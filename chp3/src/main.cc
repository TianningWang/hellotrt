#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvInferRuntime.h>

inline void check(cudaError_t call, const char* file, const int line)
{
	if (call != cudaSuccess)
	{
		printf("--- cuda error: %s at %s, line %d\n", cudaGetErrorName(call), file, line);
		printf("--- error message: %s\n", cudaGetErrorString(call));
	}
}

#define checkRuntime(call) (check(call, __FILE__, __LINE__))

inline const char* severity_string(nvinfer1::ILogger::Severity t)
{
	switch (t)
	{
		case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
			return "internal_error";
		case nvinfer1::ILogger::Severity::kERROR:
			return "error";
		case nvinfer1::ILogger::Severity::kWARNING:
			return "warning";
		case nvinfer1::ILogger::Severity::kINFO:
			return "info";
		case nvinfer1::ILogger::Severity::kVERBOSE:
			return "verbose";
		default:
			return "unknow";
	}
}

class TRTLogger : public nvinfer1::ILogger
{
public:
	virtual void log(Severity severity, const nvinfer1::AsciiChar* msg)noexcept override
	{
		if (severity <= Severity::kINFO)
		{
			if (severity == Severity::kWARNING)
			{
				printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
			}
			else if (severity == Severity::kERROR)
			{
				printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
			}
			else
			{
				printf("%s: %s\n", severity_string(severity), msg);
			}
		}
	}
}logger;

nvinfer1::Weights make_weight(float* weights_ptr, const int weights_num)
{
	nvinfer1::Weights w;
	w.count = weights_num;
	w.type = nvinfer1::DataType::kFLOAT;
	w.values = weights_ptr;
	return w;
}

bool build_model()
{
	TRTLogger logger;
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1);

	const int num_input = 1;
	const int num_output = 1;
	float layer1_weight_values[] = {
		1.0, 2.0, 3.1,
		0.1, 0.1, 0.1,
		0.2, 0.2, 0.2
	};
	float layer1_bias_values[] = {0.0};

	// set dynamic shape by profile
	nvinfer1::ITensor* input = network->addInput("image", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4(-1, num_input, -1, -1));
	nvinfer1::Weights layer1_weight  = make_weight(layer1_weight_values, 9);
	nvinfer1::Weights layer1_bias = make_weight(layer1_bias_values, 1);
	auto layer1 = network->addConvolution(*input, num_output, nvinfer1::DimsHW(3, 3), layer1_weight, layer1_bias);
	layer1->setPadding(nvinfer1::DimsHW(1, 1));
	auto prob = network->addActivation(*layer1->getOutput(0), nvinfer1::ActivationType::kRELU);
	network->markOutput(*prob->getOutput(0));

	int maxBatchSize = 10;
	printf("Workspace Size = %.2fMB\n", (1 << 28) / 1024.0f / 1024.0f);
	config->setMaxWorkspaceSize(1 << 28);
	
	// profile
	auto profile = builder->createOptimizationProfile();
	profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, num_input, 3, 3));
	profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, num_input, 3, 3));
	profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(maxBatchSize, num_input, 5, 5));
	config->addOptimizationProfile(profile);

	nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
	if (engine == nullptr)
	{
		printf("Build engine failed.\n");
		engine->destroy();
		config->destroy();
		network->destroy();
		builder->destroy();
		return false;
	}

	nvinfer1::IHostMemory* model_data = engine->serialize();
	FILE* f = fopen("/home/tiwang/code/trt/dynamic_cnn/engine/dyn.plan", "wb");
	fwrite(model_data->data(), 1, model_data->size(), f);
	fclose(f);
	model_data->destroy();
	engine->destroy();
	network->destroy();
	config->destroy();
	builder->destroy();
	printf("Done.\n");
	return true;
}

std::vector<unsigned char> load_file(const std::string& file)
{
	std::ifstream in(file, std::ios::in | std::ios::binary);
	if (!in.is_open())
	{
		return {};
	}
	in.seekg(0, std::ios::end);
	size_t length = in.tellg();
	std::vector<uint8_t> data;
	if (length > 0)
	{
		in.seekg(0, std::ios::beg);
		data.resize(length);
		in.read((char*)&data[0], length);
	}
	in.close();
	return data;
}

void inference()
{
	TRTLogger logger;
	auto engine_data = load_file("/home/tiwang/code/trt/dynamic_cnn/engine/dyn.plan");
	nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
	nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
	if (engine == nullptr)
	{
		printf("Deserialize cuda engine failed.\n");
		runtime->destroy();
		return;
	}
	nvinfer1::IExecutionContext* execution_context = engine->createExecutionContext();
	cudaStream_t stream = nullptr;
	checkRuntime(cudaStreamCreate(&stream));

	float h_input_data[] = {
		1, 1, 1,
		1, 1, 1,
		1, 1, 1,
		-1, 1, 1,
		1, 0, 1,
		1, 1, -1
	};
	float* d_input_data = nullptr;
	int ib = 2;
	int iw = 3;
	int ih = 3;
	float h_output_data[ib * iw * ih];
	float* d_output_data = nullptr;
	checkRuntime(cudaMalloc((void**)&d_input_data, sizeof(h_input_data)));
	checkRuntime(cudaMalloc((void**)&d_output_data, sizeof(h_output_data)));
	checkRuntime(cudaMemcpyAsync(d_input_data, h_input_data, sizeof(h_input_data), cudaMemcpyHostToDevice, stream));
	execution_context->setBindingDimensions(0, nvinfer1::Dims4(ib, 1, ih, iw));
	float* bindings[] = {d_input_data, d_output_data};
	bool success = execution_context->enqueueV2((void**)bindings, stream, nullptr);
	checkRuntime(cudaMemcpyAsync(h_output_data, d_output_data, sizeof(h_output_data), cudaMemcpyDeviceToHost, stream));
	checkRuntime(cudaStreamSynchronize(stream));

	// output result
	for (int b = 0; b < ib; ++b)
	{
		printf("batch %d. output_data_host = \n", b);
		for (int i = 0; i < iw * ih; ++i)
		{
			printf("%f, ", h_output_data[b * iw * ih + i]);
			if ((i + 1) % iw == 0)
			{
				printf("\n");
			}
		}
	}

	printf("Clean memory.\n");
	checkRuntime(cudaStreamDestroy(stream));
	checkRuntime(cudaFree(d_input_data));
	checkRuntime(cudaFree(d_output_data));
	execution_context->destroy();
	engine->destroy();
	runtime->destroy();
}

int main()
{
	if (!build_model())
	{
		return -1;
	}
	inference();
	return 0;
}
