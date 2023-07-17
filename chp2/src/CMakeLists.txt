CMAKE_MINIMUM_REQUIRED(VERSION 3.15)
PROJECT(infertest)

LINK_DIRECTORIES("/usr/local/TensorRT-8.5.3.1/lib")
LINK_DIRECTORIES("/usr/local/cuda-11.7/lib64")

ADD_EXECUTABLE(exec main.cc)
TARGET_LINK_LIBRARIES(exec cudart nvinfer)
