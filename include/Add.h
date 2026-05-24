#ifndef INFERENCE_ENGINE_ADD_H
#define INFERENCE_ENGINE_ADD_H

#include "Tensor.h"
#include "Ops.h"

class Add : public Ops {
public:
    Add();
    Tensor forward(std::span<Tensor> inputs) const override;
};

#endif
