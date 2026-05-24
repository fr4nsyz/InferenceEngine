#ifndef INFERENCE_ENGINE_MODEL_H
#define INFERENCE_ENGINE_MODEL_H

#include "Ops.h"
#include "Tensor.h"
#include <memory>
#include <string>
#include <vector>

class Model {
std::vector<std::unique_ptr<Ops>> _layers;

public:
explicit Model(const std::string& onnx_path);
Tensor run(const Tensor& input) const;
};

#endif
