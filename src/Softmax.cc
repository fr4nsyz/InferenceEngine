#include "../include/Softmax.h"
#include <cmath>

Softmax::Softmax() {}

Matrix Softmax::forward(std::span<Matrix> inputs) const {
  Matrix input = inputs[0];

  Matrix res = Matrix(input._rows, input._cols);

  float maxVal = input(0, 0);

  // find max
  for (int i = 0; i < input._rows; ++i) {
    for (int j = 0; j < input._cols; ++j) {
      maxVal = std::max(maxVal, input(i, j));
    }
  }

  float denominator = [&] {
    float sum = 0.0f;

    for (int i = 0; i < input._rows; ++i) {
      for (int j = 0; j < input._cols; ++j) {
        sum += std::exp(input(i, j) - maxVal);
      }
    }

    return sum;
  }();

  for (int i = 0; i < input._rows; ++i) {
    for (int j = 0; j < input._cols; ++j) {
      res(i, j) = std::exp(input(i, j) - maxVal) / denominator;
    }
  }

  return res;
};
