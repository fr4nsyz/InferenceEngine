#include "../include/Softmax.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

Softmax::Softmax(int axis) : _axis(axis) {}

Tensor Softmax::forward(std::span<Tensor> inputs) const {
  Tensor input = inputs[0];

  int axis = _axis;
  if (axis < 0) axis += input.ndim();
  if (axis < 0 || axis >= input.ndim()) {
    throw std::invalid_argument("Invalid axis for softmax");
  }

  Tensor result(input.shape());
  const float* data = input.data();
  float* res_data = result.data();
  const auto& strides = input.strides();
  const auto& shape = input.shape();
  int axis_size = shape[axis];
  int axis_stride = strides[axis];
  int total = input.size();

  for (int chunk_start = 0; chunk_start < total;
       chunk_start += axis_size * axis_stride) {
    for (int offset = 0; offset < axis_stride; ++offset) {
      int base = chunk_start + offset;

      float max_val = data[base];
      for (int k = 1; k < axis_size; ++k) {
        max_val = std::max(max_val, data[base + k * axis_stride]);
      }

      float sum = 0.0f;
      for (int k = 0; k < axis_size; ++k) {
        int idx = base + k * axis_stride;
        res_data[idx] = std::exp(data[idx] - max_val);
        sum += res_data[idx];
      }

      for (int k = 0; k < axis_size; ++k) {
        res_data[base + k * axis_stride] /= sum;
      }
    }
  }

  return result;
}
