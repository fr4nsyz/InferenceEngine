#ifndef INFERENCE_ENGINE_TENSOR_H
#define INFERENCE_ENGINE_TENSOR_H

#include <vector>
#include <initializer_list>

class Tensor {
  std::vector<float> _data;
  std::vector<int> _shape;
  std::vector<int> _strides;

  void compute_strides();

public:
  explicit Tensor(std::vector<int> shape);

  float& operator()(std::initializer_list<int> indices);
  float operator()(std::initializer_list<int> indices) const;

  int ndim() const;
  const std::vector<int>& shape() const;
  const std::vector<int>& strides() const;
  int size() const;

  Tensor operator+(const Tensor& other) const;
  Tensor operator*(const Tensor& other) const;

  Tensor reshape(std::vector<int> new_shape) const;
  Tensor transpose() const;

  float* data();
  const float* data() const;
};

#endif
