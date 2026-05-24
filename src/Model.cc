#include "../include/Model.h"
#include "../include/ONNXLoader.h"

Model::Model(const std::string& onnx_path)
: _layers(ONNXLoader::load(onnx_path)) {}

Tensor Model::run(const Tensor& input) const {
Tensor x = input;
for (const auto& layer : _layers) {
Tensor arr[] = {x};
x = layer->forward(arr);
}
return x;
}
