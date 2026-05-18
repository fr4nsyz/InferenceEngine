#include "../include/Add.h"
#include <stdexcept>

Add::Add() {};

Matrix Add::forward(std::span<Matrix> inputs) const {

  if (inputs.size() == 0) {
    throw std::invalid_argument(
        "Must provide more than one argument to Add::forward");
  }

  const bool sameShape = [&] {
    size_t firstRows = inputs[0]._rows, firstCols = inputs[0]._cols;
    for (int i = 1; i < inputs.size(); ++i) {
      if (inputs[i]._rows != firstRows or inputs[i]._cols != firstCols) {
        return false;
      }
    }
    return true;
  }();

  if (!sameShape) {
    // might implement broadcasting later, but for now we will just assert same
    // size
    throw std::invalid_argument(
        "Must provide arguments with matching shapes to Add::forward");
  }

  Matrix res = Matrix(inputs[0]._rows, inputs[0]._cols);

  for (const auto &m : inputs) {
    res = res + m;
  }

  return res;
};
