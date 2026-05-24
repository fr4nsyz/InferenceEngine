#ifndef INFERENCE_ENGINE_RELU_H
#define INFERENCE_ENGINE_RELU_H

#include "Tensor.h"
#include "Ops.h"

class ReLU : public Ops {
public:
    ReLU();
    Tensor forward(std::span<Tensor> inputs) const override;
};

#endif
