#include "../include/Linear.h"

Linear::Linear(int in_feats, int out_feats, const Matrix &weights,
               const Matrix &bias)
    : _in_feats(in_feats), _out_feats(out_feats), _weights(weights),
      _bias(bias) {}

Matrix Linear::forward(const Matrix &input) const {
  Matrix res = Matrix(1, _weights._rows);
  return _weights * input + _bias;
}
