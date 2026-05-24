cmake_minimum_required(VERSION 3.14)
project(InferenceEngine VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)

FetchContent_Declare(
    abseil-cpp
    URL https://github.com/abseil/abseil-cpp/releases/download/20240722.0/abseil-cpp-20240722.0.tar.gz
)
set(ABSL_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(ABSL_BUILD_TEST_HELPERS OFF CACHE BOOL "" FORCE)
set(ABSL_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(abseil-cpp)

FetchContent_Declare(
    protobuf
    URL https://github.com/protocolbuffers/protobuf/releases/download/v28.3/protobuf-28.3.tar.gz
)
set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(protobuf_ABSL_PROVIDER "package" CACHE STRING "" FORCE)
set(protobuf_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(protobuf)

include(${protobuf_SOURCE_DIR}/cmake/protobuf-generate.cmake)

file(DOWNLOAD
    https://raw.githubusercontent.com/onnx/onnx/v1.16.1/onnx/onnx.proto
    ${CMAKE_BINARY_DIR}/onnx/onnx.proto
)

add_library(inference_engine
src/Tensor.cc
src/Ops.cc
src/Linear.cc
src/Softmax.cc
src/ReLU.cc
src/Add.cc
src/ONNXLoader.cc
src/Model.cc
)
target_include_directories(inference_engine PUBLIC include)

protobuf_generate(
TARGET inference_engine
LANGUAGE cpp
APPEND_PATH
PROTOS ${CMAKE_BINARY_DIR}/onnx/onnx.proto
)
target_include_directories(inference_engine PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(inference_engine PUBLIC protobuf::libprotobuf)

option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
