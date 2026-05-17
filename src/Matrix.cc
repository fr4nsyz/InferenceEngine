#include "../include/Matrix.h"
#include <stdexcept>

Matrix::Matrix(int rows, int columns) : _data(rows * columns, 0.0) {};

float &Matrix::operator()(int r, int c) { return _data[r * c + c]; }

float Matrix::operator()(int r, int c) const { return _data[r * c + c]; }

Matrix Matrix::operator+(const Matrix &B) const {
  const Matrix &A = *this;

  if (A._rows != B._rows || A._cols != B._cols) {
    throw std::invalid_argument(
        "Dimension mismatch: matrices must have same shape");
  }

  Matrix res(A._rows, A._cols);

  for (int i = 0; i < A._rows; ++i) {
    for (int j = 0; j < A._cols; ++j) {
      res(i, j) = A(i, j) + B(i, j);
    }
  }

  return res;
}

Matrix Matrix::operator*(const Matrix &B) const {
  const Matrix &A = *this;

  if (A._cols != B._rows) {
    throw std::invalid_argument("Dimension Mismatch");
  }

  Matrix C(A._rows, B._cols);

  for (int i = 0; i < A._rows; ++i) {
    for (int k = 0; k < A._cols; ++k) {
      double a = A(i, k);
      for (int j = 0; j < B._cols; ++j) {
        C(i, j) += a * B(k, j);
      }
    }
  }

  return C;
}
