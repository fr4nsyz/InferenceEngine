#ifndef INFERENCE_ENGINE_ONNX_LOADER_H
#define INFERENCE_ENGINE_ONNX_LOADER_H

#include "Ops.h"
#include <memory>
#include <string>
#include <vector>

class ONNXLoader {
public:
static std::vector<std::unique_ptr<Ops>> load(const std::string& path);
};

#endif
