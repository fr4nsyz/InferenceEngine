#include "../include/ReLU.h"
#include <algorithm>

ReLU::ReLU() {}

Matrix ReLU::forward(std::span<Matrix> inputs) const {
  Matrix input = inputs[0];

  Matrix res = Matrix(input._rows, input._cols);

  for (int i = 0; i < input._rows; ++i) {

    for (int j = 0; j < input._cols; ++j) {
      res(i, j) = std::max(0.0f, input(i, j));
    }
  }

  return res;
}
