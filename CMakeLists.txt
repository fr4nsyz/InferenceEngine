cmake_minimum_required(VERSION 3.14)
project(InferenceEngine VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(inference_engine
  src/Matrix.cc
  src/Linear.cc
  src/Softmax.cc
  src/ReLU.cc
  src/Add.cc
)
target_include_directories(inference_engine PUBLIC include)

option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
