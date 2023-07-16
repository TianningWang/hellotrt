CMAKE_MINIMUM_REQUIRED(VERSION 3.15)
PROJECT(trtmnist)

LINK_DIRECTORIES("/usr/local/cuda-11.7/lib64")
LINK_DIRECTORIES("/usr/local/TensorRT-8.5.3.1/lib")

ADD_EXECUTABLE(exec trtOnnxMnist.cc)
TARGET_LINK_LIBRARIES(exec cudart nvinfer nvonnxparser)
