#include "../include/ReLU.h"
#include <algorithm>

ReLU::ReLU() {}

Tensor ReLU::forward(std::span<Tensor> inputs) const {
  Tensor input = inputs[0];
  Tensor res(input.shape());

  for (int i = 0; i < input.size(); ++i) {
    res.data()[i] = std::max(0.0f, input.data()[i]);
  }

  return res;
}
