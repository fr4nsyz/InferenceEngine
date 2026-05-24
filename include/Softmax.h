#ifndef INFERENCE_ENGINE_SOFTMAX_H
#define INFERENCE_ENGINE_SOFTMAX_H

#include "Tensor.h"
#include "Ops.h"

class Softmax : public Ops {
    int _axis;
public:
    Softmax(int axis = -1);
    Tensor forward(std::span<Tensor> inputs) const override;
};

#endif
