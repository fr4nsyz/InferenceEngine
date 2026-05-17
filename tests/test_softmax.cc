#include "../include/Softmax.h"
#include <gtest/gtest.h>

TEST(SoftmaxTest, OutputsSumToOne) {
  Softmax sm;
  Matrix input(1, 3);
  input(0, 0) = 1; input(0, 1) = 2; input(0, 2) = 3;
  Matrix out = sm.forward(input);
  float sum = 0;
  for (int i = 0; i < out._cols; ++i) sum += out(0, i);
  EXPECT_NEAR(sum, 1.0f, 1e-5);
}

TEST(SoftmaxTest, OutputsInZeroOneRange) {
  Softmax sm;
  Matrix input(1, 3);
  input(0, 0) = 1; input(0, 1) = 2; input(0, 2) = 3;
  Matrix out = sm.forward(input);
  for (int i = 0; i < out._cols; ++i) {
    EXPECT_GT(out(0, i), 0);
    EXPECT_LT(out(0, i), 1);
  }
}

TEST(SoftmaxTest, LargerInputGetsHigherProbability) {
  Softmax sm;
  Matrix input(1, 4);
  input(0, 0) = 1; input(0, 1) = 5; input(0, 2) = 2; input(0, 3) = 3;
  Matrix out = sm.forward(input);
  EXPECT_GT(out(0, 1), out(0, 3));
  EXPECT_GT(out(0, 3), out(0, 2));
  EXPECT_GT(out(0, 2), out(0, 0));
}

TEST(SoftmaxTest, UniformInputGivesUniformOutput) {
  Softmax sm;
  Matrix input(1, 4);
  for (int i = 0; i < 4; ++i) input(0, i) = 0;
  Matrix out = sm.forward(input);
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(out(0, i), 0.25f, 1e-5);
  }
}

TEST(SoftmaxTest, SingleElement) {
  Softmax sm;
  Matrix input(1, 1);
  input(0, 0) = 42;
  Matrix out = sm.forward(input);
  EXPECT_NEAR(out(0, 0), 1.0f, 1e-5);
}

TEST(SoftmaxTest, MultiDimensionalInput) {
  Softmax sm;
  Matrix input(2, 2);
  input(0, 0) = 1; input(0, 1) = 2;
  input(1, 0) = 3; input(1, 1) = 4;
  Matrix out = sm.forward(input);
  float sum = 0;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      sum += out(i, j);
  EXPECT_NEAR(sum, 1.0f, 1e-5);
}

TEST(SoftmaxTest, NonSquareInput) {
  Softmax sm;
  Matrix input(2, 3);
  input(0, 0) = 1; input(0, 1) = 2; input(0, 2) = 3;
  input(1, 0) = 0; input(1, 1) = -1; input(1, 2) = -2;
  Matrix out = sm.forward(input);
  float sum = 0;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      sum += out(i, j);
  EXPECT_NEAR(sum, 1.0f, 1e-5);
}

TEST(SoftmaxTest, OutputShapeMatchesInput) {
  Softmax sm;
  Matrix input(3, 5);
  Matrix out = sm.forward(input);
  EXPECT_EQ(out._rows, 3);
  EXPECT_EQ(out._cols, 5);
}
