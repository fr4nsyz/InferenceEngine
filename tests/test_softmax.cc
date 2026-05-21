#include "../include/Softmax.h"
#include <gtest/gtest.h>

TEST(SoftmaxTest, OutputsSumToOne) {
  Softmax sm;
  Tensor input({1, 3});
  input({0, 0}) = 1; input({0, 1}) = 2; input({0, 2}) = 3;
  Tensor out = sm.forward(input);
  float sum = 0;
  for (int i = 0; i < 3; ++i) sum += out({0, i});
  EXPECT_NEAR(sum, 1.0f, 1e-5);
}

TEST(SoftmaxTest, OutputsInZeroOneRange) {
  Softmax sm;
  Tensor input({1, 3});
  input({0, 0}) = 1; input({0, 1}) = 2; input({0, 2}) = 3;
  Tensor out = sm.forward(input);
  for (int i = 0; i < 3; ++i) {
    EXPECT_GT(out({0, i}), 0);
    EXPECT_LT(out({0, i}), 1);
  }
}

TEST(SoftmaxTest, LargerInputGetsHigherProbability) {
  Softmax sm;
  Tensor input({1, 4});
  input({0, 0}) = 1; input({0, 1}) = 5; input({0, 2}) = 2; input({0, 3}) = 3;
  Tensor out = sm.forward(input);
  EXPECT_GT(out({0, 1}), out({0, 3}));
  EXPECT_GT(out({0, 3}), out({0, 2}));
  EXPECT_GT(out({0, 2}), out({0, 0}));
}

TEST(SoftmaxTest, UniformInputGivesUniformOutput) {
  Softmax sm;
  Tensor input({1, 4});
  for (int i = 0; i < 4; ++i) input({0, i}) = 0;
  Tensor out = sm.forward(input);
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(out({0, i}), 0.25f, 1e-5);
  }
}

TEST(SoftmaxTest, SingleElement) {
  Softmax sm;
  Tensor input({1, 1});
  input({0, 0}) = 42;
  Tensor out = sm.forward(input);
  EXPECT_NEAR(out({0, 0}), 1.0f, 1e-5);
}

TEST(SoftmaxTest, MultiDimensionalInput) {
  Softmax sm;
  Tensor input({2, 2});
  input({0, 0}) = 1; input({0, 1}) = 2;
  input({1, 0}) = 3; input({1, 1}) = 4;
  Tensor out = sm.forward(input);
  for (int i = 0; i < 2; ++i) {
    float sum = out({i, 0}) + out({i, 1});
    EXPECT_NEAR(sum, 1.0f, 1e-5);
  }
}

TEST(SoftmaxTest, NonSquareInput) {
  Softmax sm;
  Tensor input({2, 3});
  input({0, 0}) = 1; input({0, 1}) = 2; input({0, 2}) = 3;
  input({1, 0}) = 0; input({1, 1}) = -1; input({1, 2}) = -2;
  Tensor out = sm.forward(input);
  for (int i = 0; i < 2; ++i) {
    float sum = 0;
    for (int j = 0; j < 3; ++j) sum += out({i, j});
    EXPECT_NEAR(sum, 1.0f, 1e-5);
  }
}

TEST(SoftmaxTest, OutputShapeMatchesInput) {
  Softmax sm;
  Tensor input({3, 5});
  Tensor out = sm.forward(input);
  EXPECT_EQ(out.shape(), std::vector<int>({3, 5}));
}

TEST(SoftmaxTest, SoftmaxAlongAxis0) {
  Softmax sm(0);
  Tensor input({3, 2});
  input({0, 0}) = 1; input({0, 1}) = 2;
  input({1, 0}) = 3; input({1, 1}) = 4;
  input({2, 0}) = 5; input({2, 1}) = 6;
  Tensor out = sm.forward(input);
  for (int j = 0; j < 2; ++j) {
    float sum = out({0, j}) + out({1, j}) + out({2, j});
    EXPECT_NEAR(sum, 1.0f, 1e-5);
  }
}

TEST(SoftmaxTest, InvalidAxis) {
  Tensor input({2, 3});
  EXPECT_THROW(Softmax(5).forward(input), std::invalid_argument);
  EXPECT_THROW(Softmax(-5).forward(input), std::invalid_argument);
}

TEST(SoftmaxTest, SoftmaxOn3DInput) {
  Softmax sm;
  Tensor input({2, 3, 4});
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 4; ++k)
        input({i, j, k}) = static_cast<float>(i + j + k);
  Tensor out = sm.forward(input);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      float sum = 0;
      for (int k = 0; k < 4; ++k) sum += out({i, j, k});
      EXPECT_NEAR(sum, 1.0f, 1e-5);
    }
  }
}
