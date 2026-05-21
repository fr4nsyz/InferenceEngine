#include "../include/Tensor.h"
#include <gtest/gtest.h>

TEST(TensorTest, Construction2D) {
  Tensor t({3, 4});
  std::vector<int> expected_shape = {3, 4};
  EXPECT_EQ(t.shape(), expected_shape);
  EXPECT_EQ(t.ndim(), 2);
  EXPECT_EQ(t.size(), 12);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_FLOAT_EQ(t({i, j}), 0.0f);
    }
  }
}

TEST(TensorTest, Construction1D) {
  Tensor t({5});
  std::vector<int> expected_shape = {5};
  EXPECT_EQ(t.shape(), expected_shape);
  EXPECT_EQ(t.ndim(), 1);
  EXPECT_EQ(t.size(), 5);
  for (int i = 0; i < 5; ++i) {
    EXPECT_FLOAT_EQ(t({i}), 0.0f);
  }
}

TEST(TensorTest, Construction3D) {
  Tensor t({2, 3, 4});
  std::vector<int> expected_shape = {2, 3, 4};
  EXPECT_EQ(t.shape(), expected_shape);
  EXPECT_EQ(t.ndim(), 3);
  EXPECT_EQ(t.size(), 24);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 4; ++k)
        EXPECT_FLOAT_EQ(t({i, j, k}), 0.0f);
}

TEST(TensorTest, ConstructionEmptyShape) {
  EXPECT_THROW(Tensor({}), std::invalid_argument);
}

TEST(TensorTest, ConstructionInvalidDimension) {
  EXPECT_THROW(Tensor({2, -1, 3}), std::invalid_argument);
  EXPECT_THROW(Tensor({0}), std::invalid_argument);
}

TEST(TensorTest, ElementAccessReadWrite) {
  Tensor t({2, 3});
  t({0, 1}) = 5.0f;
  t({1, 2}) = -3.5f;
  EXPECT_FLOAT_EQ(t({0, 1}), 5.0f);
  EXPECT_FLOAT_EQ(t({1, 2}), -3.5f);
  EXPECT_FLOAT_EQ(t({0, 0}), 0.0f);
}

TEST(TensorTest, ElementAccess3D) {
  Tensor t({2, 3, 4});
  t({1, 2, 3}) = 7.5f;
  EXPECT_FLOAT_EQ(t({1, 2, 3}), 7.5f);
  EXPECT_FLOAT_EQ(t({0, 0, 0}), 0.0f);
}

TEST(TensorTest, ElementAccessOutOfBounds) {
  Tensor t({2, 3});
  EXPECT_THROW(t({2, 0}), std::out_of_range);
  EXPECT_THROW(t({0, 3}), std::out_of_range);
  EXPECT_THROW(t({-1, 0}), std::out_of_range);
}

TEST(TensorTest, WrongNumberOfIndices) {
  Tensor t({2, 3});
  EXPECT_THROW(t({1}), std::invalid_argument);
  EXPECT_THROW(t({1, 2, 3}), std::invalid_argument);
}

TEST(TensorTest, Addition) {
  Tensor A({2, 2});
  A({0, 0}) = 1; A({0, 1}) = 2;
  A({1, 0}) = 3; A({1, 1}) = 4;

  Tensor B({2, 2});
  B({0, 0}) = 5; B({0, 1}) = 6;
  B({1, 0}) = 7; B({1, 1}) = 8;

  Tensor C = A + B;
  EXPECT_FLOAT_EQ(C({0, 0}), 6);
  EXPECT_FLOAT_EQ(C({0, 1}), 8);
  EXPECT_FLOAT_EQ(C({1, 0}), 10);
  EXPECT_FLOAT_EQ(C({1, 1}), 12);
}

TEST(TensorTest, Addition3D) {
  Tensor A({2, 2, 2});
  A({0, 0, 0}) = 1; A({0, 0, 1}) = 2;
  A({0, 1, 0}) = 3; A({0, 1, 1}) = 4;
  A({1, 0, 0}) = 5; A({1, 0, 1}) = 6;
  A({1, 1, 0}) = 7; A({1, 1, 1}) = 8;

  Tensor B({2, 2, 2});
  B({0, 0, 0}) = 8; B({0, 0, 1}) = 7;
  B({0, 1, 0}) = 6; B({0, 1, 1}) = 5;
  B({1, 0, 0}) = 4; B({1, 0, 1}) = 3;
  B({1, 1, 0}) = 2; B({1, 1, 1}) = 1;

  Tensor C = A + B;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      for (int k = 0; k < 2; ++k)
        EXPECT_FLOAT_EQ(C({i, j, k}), 9);
}

TEST(TensorTest, AdditionBroadcast1DTo2D) {
  Tensor A({2, 3});
  A({0, 0}) = 1; A({0, 1}) = 2; A({0, 2}) = 3;
  A({1, 0}) = 4; A({1, 1}) = 5; A({1, 2}) = 6;

  Tensor B({3});
  B({0}) = 10; B({1}) = 20; B({2}) = 30;

  Tensor C = A + B;
  EXPECT_FLOAT_EQ(C({0, 0}), 11);
  EXPECT_FLOAT_EQ(C({0, 1}), 22);
  EXPECT_FLOAT_EQ(C({0, 2}), 33);
  EXPECT_FLOAT_EQ(C({1, 0}), 14);
  EXPECT_FLOAT_EQ(C({1, 1}), 25);
  EXPECT_FLOAT_EQ(C({1, 2}), 36);
}

TEST(TensorTest, AdditionBroadcastWithScalarDim) {
  Tensor A({2, 3});
  A({0, 0}) = 1; A({0, 1}) = 2; A({0, 2}) = 3;
  A({1, 0}) = 4; A({1, 1}) = 5; A({1, 2}) = 6;

  Tensor B({1, 3});
  B({0, 0}) = 10; B({0, 1}) = 20; B({0, 2}) = 30;

  Tensor C = A + B;
  EXPECT_FLOAT_EQ(C({0, 0}), 11);
  EXPECT_FLOAT_EQ(C({1, 1}), 25);
  EXPECT_FLOAT_EQ(C({1, 2}), 36);
}

TEST(TensorTest, AdditionDimensionMismatch) {
  Tensor A({2, 3});
  Tensor B({3, 2});
  EXPECT_THROW(A + B, std::invalid_argument);
}

TEST(TensorTest, Multiplication) {
  Tensor A({2, 3});
  A({0, 0}) = 1; A({0, 1}) = 2; A({0, 2}) = 3;
  A({1, 0}) = 4; A({1, 1}) = 5; A({1, 2}) = 6;

  Tensor B({3, 2});
  B({0, 0}) = 7;  B({0, 1}) = 8;
  B({1, 0}) = 9;  B({1, 1}) = 10;
  B({2, 0}) = 11; B({2, 1}) = 12;

  Tensor C = A * B;
  std::vector<int> expected_shape = {2, 2};
  EXPECT_EQ(C.shape(), expected_shape);
  EXPECT_FLOAT_EQ(C({0, 0}), 58);
  EXPECT_FLOAT_EQ(C({0, 1}), 64);
  EXPECT_FLOAT_EQ(C({1, 0}), 139);
  EXPECT_FLOAT_EQ(C({1, 1}), 154);
}

TEST(TensorTest, MultiplicationNonSquare) {
  Tensor A({1, 4});
  A({0, 0}) = 1; A({0, 1}) = 2; A({0, 2}) = 3; A({0, 3}) = 4;

  Tensor B({4, 1});
  B({0, 0}) = 5;
  B({1, 0}) = 6;
  B({2, 0}) = 7;
  B({3, 0}) = 8;

  Tensor C = A * B;
  EXPECT_EQ(C.shape(), std::vector<int>({1, 1}));
  EXPECT_FLOAT_EQ(C({0, 0}), 70);
}

TEST(TensorTest, MultiplicationDimensionMismatch) {
  Tensor A({2, 2});
  Tensor B({3, 3});
  EXPECT_THROW(A * B, std::invalid_argument);
}

TEST(TensorTest, MultiplicationNon2D) {
  Tensor A({2, 2, 2});
  Tensor B({2, 2, 2});
  EXPECT_THROW(A * B, std::invalid_argument);
}

TEST(TensorTest, NonSquareElementAccess) {
  Tensor t({3, 5});
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 5; ++j) {
      t({i, j}) = static_cast<float>(i * 5 + j);
    }
  }
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 5; ++j) {
      EXPECT_FLOAT_EQ(t({i, j}), static_cast<float>(i * 5 + j));
    }
  }
}

TEST(TensorTest, AdditionNonSquare) {
  Tensor A({2, 3});
  A({0, 0}) = 1; A({0, 1}) = 2; A({0, 2}) = 3;
  A({1, 0}) = 4; A({1, 1}) = 5; A({1, 2}) = 6;

  Tensor B({2, 3});
  B({0, 0}) = 6; B({0, 1}) = 5; B({0, 2}) = 4;
  B({1, 0}) = 3; B({1, 1}) = 2; B({1, 2}) = 1;

  Tensor C = A + B;
  EXPECT_FLOAT_EQ(C({0, 0}), 7);
  EXPECT_FLOAT_EQ(C({0, 1}), 7);
  EXPECT_FLOAT_EQ(C({0, 2}), 7);
  EXPECT_FLOAT_EQ(C({1, 0}), 7);
  EXPECT_FLOAT_EQ(C({1, 1}), 7);
  EXPECT_FLOAT_EQ(C({1, 2}), 7);
}

TEST(TensorTest, IdentityMultiplication) {
  Tensor A({3, 3});
  A({0, 0}) = 1; A({0, 1}) = 2; A({0, 2}) = 3;
  A({1, 0}) = 4; A({1, 1}) = 5; A({1, 2}) = 6;
  A({2, 0}) = 7; A({2, 1}) = 8; A({2, 2}) = 9;

  Tensor I({3, 3});
  I({0, 0}) = 1; I({0, 1}) = 0; I({0, 2}) = 0;
  I({1, 0}) = 0; I({1, 1}) = 1; I({1, 2}) = 0;
  I({2, 0}) = 0; I({2, 1}) = 0; I({2, 2}) = 1;

  Tensor C = A * I;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_FLOAT_EQ(C({i, j}), A({i, j}));
    }
  }
}

TEST(TensorTest, OneByOne) {
  Tensor A({1, 1});
  A({0, 0}) = 42;
  Tensor B({1, 1});
  B({0, 0}) = 2;

  EXPECT_FLOAT_EQ((A + B)({0, 0}), 44);
  EXPECT_FLOAT_EQ((A * B)({0, 0}), 84);
}

TEST(TensorTest, ConstElementAccess) {
  const Tensor t({2, 2});
  EXPECT_FLOAT_EQ(t({0, 0}), 0.0f);
  EXPECT_FLOAT_EQ(t({1, 1}), 0.0f);
}

TEST(TensorTest, Reshape) {
  Tensor t({2, 6});
  for (int i = 0; i < 12; ++i) t.data()[i] = static_cast<float>(i);

  Tensor r = t.reshape({3, 4});
  EXPECT_EQ(r.shape(), std::vector<int>({3, 4}));
  EXPECT_EQ(r.size(), 12);
  for (int i = 0; i < 12; ++i) {
    EXPECT_FLOAT_EQ(r.data()[i], static_cast<float>(i));
  }
}

TEST(TensorTest, ReshapeTo3D) {
  Tensor t({12});
  for (int i = 0; i < 12; ++i) t.data()[i] = static_cast<float>(i);

  Tensor r = t.reshape({2, 3, 2});
  EXPECT_EQ(r.shape(), std::vector<int>({2, 3, 2}));
  EXPECT_FLOAT_EQ(r({1, 2, 1}), 11);
}

TEST(TensorTest, ReshapeMismatch) {
  Tensor t({2, 3});
  EXPECT_THROW(t.reshape({4, 4}), std::invalid_argument);
}

TEST(TensorTest, Transpose) {
  Tensor t({2, 3});
  t({0, 0}) = 1; t({0, 1}) = 2; t({0, 2}) = 3;
  t({1, 0}) = 4; t({1, 1}) = 5; t({1, 2}) = 6;

  Tensor tr = t.transpose();
  EXPECT_EQ(tr.shape(), std::vector<int>({3, 2}));
  EXPECT_FLOAT_EQ(tr({0, 0}), 1);
  EXPECT_FLOAT_EQ(tr({1, 0}), 2);
  EXPECT_FLOAT_EQ(tr({2, 0}), 3);
  EXPECT_FLOAT_EQ(tr({0, 1}), 4);
  EXPECT_FLOAT_EQ(tr({1, 1}), 5);
  EXPECT_FLOAT_EQ(tr({2, 1}), 6);
}

TEST(TensorTest, TransposeNon2D) {
  Tensor t({2, 3, 4});
  EXPECT_THROW(t.transpose(), std::invalid_argument);
}

TEST(TensorTest, StridesAccessor) {
  Tensor t({2, 3, 4});
  auto s = t.strides();
  ASSERT_EQ(s.size(), 3);
  EXPECT_EQ(s[0], 12);
  EXPECT_EQ(s[1], 4);
  EXPECT_EQ(s[2], 1);
}

TEST(TensorTest, DataPointer) {
  Tensor t({3, 4});
  float* ptr = t.data();
  ptr[0] = 1.5f;
  EXPECT_FLOAT_EQ(t({0, 0}), 1.5f);
}
