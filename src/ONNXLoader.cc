#include "../include/ONNXLoader.h"
#include "../include/Linear.h"
#include "../include/ReLU.h"
#include "../include/Softmax.h"
#include "../include/Add.h"
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <cstring>
#include <map>
#include <onnx.pb.h>

static Tensor proto_to_tensor(const onnx::TensorProto& tp) {
    std::vector<int> shape;
    for (auto d : tp.dims()) {
        shape.push_back(static_cast<int>(d));
    }
    if (shape.empty()) {
        shape.push_back(1);
    }

    std::vector<float> data;
    if (tp.data_type() == onnx::TensorProto::FLOAT) {
        if (tp.has_raw_data()) {
            const std::string& raw = tp.raw_data();
            size_t count = raw.size() / sizeof(float);
            data.resize(count);
            std::memcpy(data.data(), raw.data(), raw.size());
        } else {
            data.assign(tp.float_data().begin(), tp.float_data().end());
        }
    } else {
        throw std::runtime_error("ONNXLoader: unsupported tensor data type: " +
            std::to_string(tp.data_type()));
    }

    return Tensor(shape, std::move(data));
}

static int get_int_attr(const onnx::NodeProto& node, const std::string& name, int default_val) {
    for (const auto& attr : node.attribute()) {
        if (attr.name() == name && attr.type() == onnx::AttributeProto::INT) {
            return static_cast<int>(attr.i());
        }
    }
    return default_val;
}

static float get_float_attr(const onnx::NodeProto& node, const std::string& name, float default_val) {
    for (const auto& attr : node.attribute()) {
        if (attr.name() == name && attr.type() == onnx::AttributeProto::FLOAT) {
            return attr.f();
        }
    }
    return default_val;
}

std::vector<std::unique_ptr<Ops>> ONNXLoader::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("ONNXLoader: cannot open file: " + path);
    }

    std::stringstream buf;
    buf << file.rdbuf();
    std::string content = buf.str();

    onnx::ModelProto model;
    if (!model.ParseFromString(content)) {
        throw std::runtime_error("ONNXLoader: failed to parse ONNX model");
    }

    const onnx::GraphProto& graph = model.graph();

    std::map<std::string, Tensor> initializers;
    for (const auto& init : graph.initializer()) {
        initializers.insert_or_assign(init.name(), proto_to_tensor(init));
    }

    std::vector<std::unique_ptr<Ops>> layers;

    for (const auto& node : graph.node()) {
        const std::string& op_type = node.op_type();

        if (op_type == "Gemm") {
            int transB = get_int_attr(node, "transB", 0);

            const std::string& weight_name = node.input(1);
            const std::string& bias_name = node.input_size() > 2 ? node.input(2) : "";

            auto wit = initializers.find(weight_name);
            if (wit == initializers.end()) {
                throw std::runtime_error("ONNXLoader: Gemm weight '" + weight_name + "' not found");
            }

            Tensor weight = wit->second;
            if (transB) {
                weight = weight.transpose();
            }

            Tensor bias({weight.shape()[1]}, std::vector<float>(weight.shape()[1], 0.0f));
            if (!bias_name.empty()) {
                auto bit = initializers.find(bias_name);
                if (bit != initializers.end()) {
                    bias = bit->second;
                }
            }

            int in_feats = weight.shape()[0];
            int out_feats = weight.shape()[1];

            layers.push_back(std::make_unique<Linear>(in_feats, out_feats, weight, bias));

        } else if (op_type == "Relu") {
            layers.push_back(std::make_unique<ReLU>());

        } else if (op_type == "Softmax") {
            int axis = get_int_attr(node, "axis", -1);
            layers.push_back(std::make_unique<Softmax>(axis));

        } else if (op_type == "Add") {
            layers.push_back(std::make_unique<Add>());

        } else {
            throw std::runtime_error("ONNXLoader: unsupported op type: " + op_type);
        }
    }

    return layers;
}
