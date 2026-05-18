#include "../include/Linear.h"

Linear::Linear(int in_feats, int out_feats, const Matrix &weights,
               const Matrix &bias)
    : _in_feats(in_feats), _out_feats(out_feats), _weights(weights),
      _bias(bias) {}

Matrix Linear::forward(std::span<Matrix> inputs) const {
  Matrix input = inputs[0];
  Matrix res = Matrix(1, _weights._rows);
  return _weights * input + _bias;
}
