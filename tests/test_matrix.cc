#include "../include/Matrix.h"
#include <gtest/gtest.h>

TEST(MatrixTest, Construction) {
  Matrix m(3, 4);
  EXPECT_EQ(m._rows, 3);
  EXPECT_EQ(m._cols, 4);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_FLOAT_EQ(m(i, j), 0.0f);
    }
  }
}

TEST(MatrixTest, ElementAccessReadWrite) {
  Matrix m(2, 3);
  m(0, 1) = 5.0f;
  m(1, 2) = -3.5f;
  EXPECT_FLOAT_EQ(m(0, 1), 5.0f);
  EXPECT_FLOAT_EQ(m(1, 2), -3.5f);
  EXPECT_FLOAT_EQ(m(0, 0), 0.0f);
}

TEST(MatrixTest, Addition) {
  Matrix A(2, 2);
  A(0, 0) = 1; A(0, 1) = 2;
  A(1, 0) = 3; A(1, 1) = 4;

  Matrix B(2, 2);
  B(0, 0) = 5; B(0, 1) = 6;
  B(1, 0) = 7; B(1, 1) = 8;

  Matrix C = A + B;
  EXPECT_FLOAT_EQ(C(0, 0), 6);
  EXPECT_FLOAT_EQ(C(0, 1), 8);
  EXPECT_FLOAT_EQ(C(1, 0), 10);
  EXPECT_FLOAT_EQ(C(1, 1), 12);
}

TEST(MatrixTest, AdditionDimensionMismatch) {
  Matrix A(2, 3);
  Matrix B(3, 2);
  EXPECT_THROW(A + B, std::invalid_argument);
}

TEST(MatrixTest, Multiplication) {
  Matrix A(2, 3);
  A(0, 0) = 1; A(0, 1) = 2; A(0, 2) = 3;
  A(1, 0) = 4; A(1, 1) = 5; A(1, 2) = 6;

  Matrix B(3, 2);
  B(0, 0) = 7;  B(0, 1) = 8;
  B(1, 0) = 9;  B(1, 1) = 10;
  B(2, 0) = 11; B(2, 1) = 12;

  Matrix C = A * B;
  EXPECT_EQ(C._rows, 2);
  EXPECT_EQ(C._cols, 2);
  EXPECT_FLOAT_EQ(C(0, 0), 58);
  EXPECT_FLOAT_EQ(C(0, 1), 64);
  EXPECT_FLOAT_EQ(C(1, 0), 139);
  EXPECT_FLOAT_EQ(C(1, 1), 154);
}

TEST(MatrixTest, MultiplicationNonSquare) {
  Matrix A(1, 4);
  A(0, 0) = 1; A(0, 1) = 2; A(0, 2) = 3; A(0, 3) = 4;

  Matrix B(4, 1);
  B(0, 0) = 5;
  B(1, 0) = 6;
  B(2, 0) = 7;
  B(3, 0) = 8;

  Matrix C = A * B;
  EXPECT_EQ(C._rows, 1);
  EXPECT_EQ(C._cols, 1);
  EXPECT_FLOAT_EQ(C(0, 0), 70);
}

TEST(MatrixTest, MultiplicationDimensionMismatch) {
  Matrix A(2, 2);
  Matrix B(3, 3);
  EXPECT_THROW(A * B, std::invalid_argument);
}

TEST(MatrixTest, NonSquareElementAccess) {
  Matrix m(3, 5);
  for (int i = 0; i < m._rows; ++i) {
    for (int j = 0; j < m._cols; ++j) {
      m(i, j) = static_cast<float>(i * m._cols + j);
    }
  }
  for (int i = 0; i < m._rows; ++i) {
    for (int j = 0; j < m._cols; ++j) {
      EXPECT_FLOAT_EQ(m(i, j), static_cast<float>(i * m._cols + j));
    }
  }
}

TEST(MatrixTest, AdditionNonSquare) {
  Matrix A(2, 3);
  A(0, 0) = 1; A(0, 1) = 2; A(0, 2) = 3;
  A(1, 0) = 4; A(1, 1) = 5; A(1, 2) = 6;

  Matrix B(2, 3);
  B(0, 0) = 6; B(0, 1) = 5; B(0, 2) = 4;
  B(1, 0) = 3; B(1, 1) = 2; B(1, 2) = 1;

  Matrix C = A + B;
  EXPECT_FLOAT_EQ(C(0, 0), 7);
  EXPECT_FLOAT_EQ(C(0, 1), 7);
  EXPECT_FLOAT_EQ(C(0, 2), 7);
  EXPECT_FLOAT_EQ(C(1, 0), 7);
  EXPECT_FLOAT_EQ(C(1, 1), 7);
  EXPECT_FLOAT_EQ(C(1, 2), 7);
}

TEST(MatrixTest, IdentityMultiplication) {
  Matrix A(3, 3);
  A(0, 0) = 1; A(0, 1) = 2; A(0, 2) = 3;
  A(1, 0) = 4; A(1, 1) = 5; A(1, 2) = 6;
  A(2, 0) = 7; A(2, 1) = 8; A(2, 2) = 9;

  Matrix I(3, 3);
  I(0, 0) = 1; I(0, 1) = 0; I(0, 2) = 0;
  I(1, 0) = 0; I(1, 1) = 1; I(1, 2) = 0;
  I(2, 0) = 0; I(2, 1) = 0; I(2, 2) = 1;

  Matrix C = A * I;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_FLOAT_EQ(C(i, j), A(i, j));
    }
  }
}

TEST(MatrixTest, OneByOne) {
  Matrix A(1, 1);
  A(0, 0) = 42;
  Matrix B(1, 1);
  B(0, 0) = 2;

  EXPECT_FLOAT_EQ((A + B)(0, 0), 44);
  EXPECT_FLOAT_EQ((A * B)(0, 0), 84);
}

TEST(MatrixTest, ConstElementAccess) {
  const Matrix m(2, 2);
  EXPECT_FLOAT_EQ(m(0, 0), 0.0f);
  EXPECT_FLOAT_EQ(m(1, 1), 0.0f);
}
