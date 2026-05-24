#ifndef INFERENCE_ENGINE_LINEAR_H
#define INFERENCE_ENGINE_LINEAR_H

#include "Tensor.h"
#include "Ops.h"

class Linear : public Ops {
    const Tensor _weights;
    const Tensor _bias;
    int _in_feats;
    int _out_feats;

public:
    Linear(int in_feats, int out_feats, const Tensor &weights,
           const Tensor &bias);
    Tensor forward(std::span<Tensor> inputs) const override;
};

#endif
