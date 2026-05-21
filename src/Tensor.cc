#include "../include/Tensor.h"
#include <stdexcept>
#include <algorithm>
#include <numeric>

Tensor::Tensor(std::vector<int> shape) : _shape(std::move(shape)) {
  if (_shape.empty()) {
    throw std::invalid_argument("Tensor must have at least 1 dimension");
  }
  for (int d : _shape) {
    if (d <= 0) {
      throw std::invalid_argument("Tensor dimensions must be positive");
    }
  }
  compute_strides();
  _data.resize(size(), 0.0f);
}

void Tensor::compute_strides() {
  int n = ndim();
  _strides.resize(n);
  if (n > 0) {
    _strides[n - 1] = 1;
    for (int i = n - 2; i >= 0; --i) {
      _strides[i] = _strides[i + 1] * _shape[i + 1];
    }
  }
}

int Tensor::ndim() const { return static_cast<int>(_shape.size()); }
const std::vector<int>& Tensor::shape() const { return _shape; }
const std::vector<int>& Tensor::strides() const { return _strides; }
int Tensor::size() const {
  if (_shape.empty()) return 0;
  return std::accumulate(_shape.begin(), _shape.end(), 1,
                         std::multiplies<int>());
}

float* Tensor::data() { return _data.data(); }
const float* Tensor::data() const { return _data.data(); }

float& Tensor::operator()(std::initializer_list<int> indices) {
  if (indices.size() != _shape.size()) {
    throw std::invalid_argument("Wrong number of indices");
  }
  int flat = 0;
  int dim = 0;
  for (int idx : indices) {
    if (idx < 0 || idx >= _shape[dim]) {
      throw std::out_of_range("Index out of bounds");
    }
    flat += idx * _strides[dim];
    ++dim;
  }
  return _data[flat];
}

float Tensor::operator()(std::initializer_list<int> indices) const {
  if (indices.size() != _shape.size()) {
    throw std::invalid_argument("Wrong number of indices");
  }
  int flat = 0;
  int dim = 0;
  for (int idx : indices) {
    if (idx < 0 || idx >= _shape[dim]) {
      throw std::out_of_range("Index out of bounds");
    }
    flat += idx * _strides[dim];
    ++dim;
  }
  return _data[flat];
}

Tensor Tensor::operator+(const Tensor& other) const {
  int max_ndim = std::max(ndim(), other.ndim());
  std::vector<int> result_shape(max_ndim);

  for (int i = 0; i < max_ndim; ++i) {
    int d1 = (i < max_ndim - ndim()) ? 1
                                     : _shape[i - (max_ndim - ndim())];
    int d2 = (i < max_ndim - other.ndim())
                 ? 1
                 : other._shape[i - (max_ndim - other.ndim())];

    if (d1 == d2) {
      result_shape[i] = d1;
    } else if (d1 == 1) {
      result_shape[i] = d2;
    } else if (d2 == 1) {
      result_shape[i] = d1;
    } else {
      throw std::invalid_argument(
          "Shapes cannot be broadcast for addition");
    }
  }

  Tensor result(result_shape);

  for (int flat = 0; flat < result.size(); ++flat) {
    int remaining = flat;
    std::vector<int> result_indices(max_ndim);
    for (int i = 0; i < max_ndim; ++i) {
      result_indices[i] = remaining / result._strides[i];
      remaining %= result._strides[i];
    }

    int this_flat = 0;
    if (ndim() > 0) {
      int offset = max_ndim - ndim();
      for (int i = 0; i < ndim(); ++i) {
        int ri = result_indices[offset + i];
        this_flat += ((_shape[i] == 1) ? 0 : ri) * _strides[i];
      }
    }

    int other_flat = 0;
    if (other.ndim() > 0) {
      int offset = max_ndim - other.ndim();
      for (int i = 0; i < other.ndim(); ++i) {
        int ri = result_indices[offset + i];
        other_flat += ((other._shape[i] == 1) ? 0 : ri) * other._strides[i];
      }
    }

    result._data[flat] = _data[this_flat] + other._data[other_flat];
  }

  return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
  if (ndim() != 2 || other.ndim() != 2) {
    throw std::invalid_argument(
        "Matrix multiply only supported for 2D tensors");
  }

  int m = _shape[0];
  int k = _shape[1];
  int n = other._shape[1];

  if (k != other._shape[0]) {
    throw std::invalid_argument("Dimension mismatch for matrix multiply");
  }

  Tensor result({m, n});

  for (int i = 0; i < m; ++i) {
    for (int kk = 0; kk < k; ++kk) {
      float a = _data[i * _strides[0] + kk * _strides[1]];
      for (int j = 0; j < n; ++j) {
        result._data[i * result._strides[0] + j * result._strides[1]] +=
            a * other._data[kk * other._strides[0] + j * other._strides[1]];
      }
    }
  }

  return result;
}

Tensor Tensor::reshape(std::vector<int> new_shape) const {
  int new_size = 1;
  for (int d : new_shape) {
    if (d <= 0) {
      throw std::invalid_argument("New shape dimensions must be positive");
    }
    new_size *= d;
  }
  if (new_size != size()) {
    throw std::invalid_argument(
        "New shape must have same number of elements");
  }
  Tensor result(new_shape);
  std::copy(_data.begin(), _data.end(), result._data.begin());
  return result;
}

Tensor Tensor::transpose() const {
  if (ndim() != 2) {
    throw std::invalid_argument(
        "Transpose only supported for 2D tensors");
  }
  Tensor result({_shape[1], _shape[0]});
  for (int i = 0; i < _shape[0]; ++i) {
    for (int j = 0; j < _shape[1]; ++j) {
      result({j, i}) = (*this)({i, j});
    }
  }
  return result;
}


