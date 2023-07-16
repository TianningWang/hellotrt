#include <stdio.h>

#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvInferRuntime.h>

inline const char* severity_string(nvinfer1::ILogger::Severity t)
{
	switch(t)
	{
		case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
			return "internal_error";
		case nvinfer1::ILogger::Severity::kERROR:
			return "error";
		case nvinfer1::ILogger::Severity::kWARNING:
			return "warining";
		case nvinfer1::ILogger::Severity::kINFO:
			return "info";
		case nvinfer1::ILogger::Severity::kVERBOSE:
			return "verbose";
		default:
			return "unknown";
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
			else if (severity <= Severity::kERROR)
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

nvinfer1::Weights make_weights(float* weight_ptr, int weight_count)
{
	nvinfer1::Weights w;
	w.count = weight_count;
	w.type = nvinfer1::DataType::kFLOAT;
	w.values = weight_ptr;
	return w;
}

bool buildModel()
{
	TRTLogger logger;
	// create builder, and then create config and network by builder
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1);

	// construct a simple network
	const int num_input = 3;
	const int num_output = 2;
	float layer1_weight_values[] = {1.0, 2.0, 0.5, 0.1, 0.2, 0.5};
	float layer1_bias_values[] = {0.3, 0.8};

	nvinfer1::ITensor* input = network->addInput("image", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4(1, num_input, 1, 1));
	nvinfer1::Weights layer1_weight = make_weights(layer1_weight_values, 6);
	nvinfer1::Weights layer1_bias = make_weights(layer1_bias_values, 2);
	auto layer1 = network->addFullyConnected(*input, num_output, layer1_weight, layer1_bias);
	auto prob = network->addActivation(*layer1->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
	network->markOutput(*prob->getOutput(0));

	printf("Workspace size = %.2f\n", (1 << 28) / 1024.0f / 1024.0f);
	config->setMaxWorkspaceSize(1 << 28);
	builder->setMaxBatchSize(1);

	nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
	if (engine == nullptr)
	{
		printf("Build engine failed.\n");
		engine->destroy();
		network->destroy();
		config->destroy();
		builder->destroy();
		return false;
	}

	nvinfer1::IHostMemory* model_data = engine->serialize();
	FILE* f = fopen("/home/tiwang/code/trt/hello_trt/engine/engine.plan", "wb");
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

int main()
{
	buildModel();
}
