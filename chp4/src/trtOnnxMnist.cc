#include <vector>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>

#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>

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
				printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
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

bool build_model()
{
	TRTLogger logger;
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1);

	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
	if (!parser->parseFromFile("/home/tiwang/code/trt/mnist/model/mnist_dynamic2.onnx", 1))
	{
		printf("failed to parser mnist.onnx\n");
		parser->destroy();
		network->destroy();
		config->destroy();
		builder->destroy();
		return false;
	}

	int max_batchsize = 256;
	printf("Workspace size = %.2fMB\n", (1 << 28) / 1024.0f / 1024.0f);
	config->setMaxWorkspaceSize(1 << 28);

	auto profile = builder->createOptimizationProfile();
	auto input_tensor = network->getInput(0);
	int input_channel = input_tensor->getDimensions().d[3];
	/*
	std::cout << input_tensor->getDimensions().d[0] << std::endl;
	std::cout << input_tensor->getDimensions().d[1] << std::endl;
	std::cout << input_tensor->getDimensions().d[2] << std::endl;
	std::cout << input_tensor->getDimensions().d[3] << std::endl;
	*/
	profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 28, 28, 1));
	profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(32, 28, 28, 1));
	profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(max_batchsize, 28, 28, 1));
	config->addOptimizationProfile(profile);
	nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
	if (engine == nullptr)
	{
		printf("Build engine failed.\n");
		parser->destroy();
		network->destroy();
		config->destroy();
		builder->destroy();
		return false;
	}

	nvinfer1::IHostMemory* model_data = engine->serialize();
	FILE* f = fopen("/home/tiwang/code/trt/mnist/engine/mnist.plan", "wb");
	fwrite(model_data->data(), 1, model_data->size(), f);
	fclose(f);

	model_data->destroy();
	parser->destroy();
	engine->destroy();
	network->destroy();
	config->destroy();
	builder->destroy();
	printf("Done.\n");
	return true;
}

int main()
{
	build_model();
}
