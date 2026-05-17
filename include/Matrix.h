#ifndef INFERENCE_ENGINE_MATRIX_IMPLEMENTATION_H
#define INFERENCE_ENGINE_MATRIX_IMPLEMENTATION_H
#include <array>
#include <tuple>
#include <vector>

class Matrix {
  std::vector<float> _data;

public:
  int _rows;
  int _cols;

  Matrix(int rows, int cols);
  float &operator()(int r, int c);
  float operator()(int r, int c) const;
  Matrix operator*(const Matrix &B) const;
  Matrix operator+(const Matrix &B) const;
};
#endif
