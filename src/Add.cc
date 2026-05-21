#include "../include/Add.h"
#include <stdexcept>

Add::Add() {};

Tensor Add::forward(std::span<Tensor> inputs) const {
  if (inputs.size() == 0) {
    throw std::invalid_argument(
        "Must provide more than one argument to Add::forward");
  }

  Tensor res = inputs[0];

  for (int i = 1; i < inputs.size(); ++i) {
    res = res + inputs[i];
  }

  return res;
};
