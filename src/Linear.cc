#include "../include/Linear.h"

Linear::Linear(int in_feats, int out_feats, const Tensor &weights,
               const Tensor &bias)
    : _in_feats(in_feats), _out_feats(out_feats), _weights(weights),
      _bias(bias) {}

Tensor Linear::forward(std::span<Tensor> inputs) const {
  Tensor input = inputs[0];
  return _weights * input + _bias;
}
