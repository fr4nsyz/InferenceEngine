#include "../include/ReLU.h"
#include <gtest/gtest.h>

TEST(ReLUTest, PositiveValuesPassThrough) {
  ReLU relu;
  Tensor input({1, 3});
  input({0, 0}) = 1; input({0, 1}) = 2.5; input({0, 2}) = 3;
  Tensor out = relu.forward(input);
  EXPECT_FLOAT_EQ(out({0, 0}), 1);
  EXPECT_FLOAT_EQ(out({0, 1}), 2.5);
  EXPECT_FLOAT_EQ(out({0, 2}), 3);
}

TEST(ReLUTest, NegativeBecomeZero) {
  ReLU relu;
  Tensor input({1, 3});
  input({0, 0}) = -1; input({0, 1}) = -2.5; input({0, 2}) = -3;
  Tensor out = relu.forward(input);
  EXPECT_FLOAT_EQ(out({0, 0}), 0);
  EXPECT_FLOAT_EQ(out({0, 1}), 0);
  EXPECT_FLOAT_EQ(out({0, 2}), 0);
}

TEST(ReLUTest, ZeroStaysZero) {
  ReLU relu;
  Tensor input({1, 2});
  input({0, 0}) = 0; input({0, 1}) = 0;
  Tensor out = relu.forward(input);
  EXPECT_FLOAT_EQ(out({0, 0}), 0);
  EXPECT_FLOAT_EQ(out({0, 1}), 0);
}

TEST(ReLUTest, MixedValues) {
  ReLU relu;
  Tensor input({2, 2});
  input({0, 0}) = -1; input({0, 1}) = 0;
  input({1, 0}) = 5;  input({1, 1}) = -3;
  Tensor out = relu.forward(input);
  EXPECT_FLOAT_EQ(out({0, 0}), 0);
  EXPECT_FLOAT_EQ(out({0, 1}), 0);
  EXPECT_FLOAT_EQ(out({1, 0}), 5);
  EXPECT_FLOAT_EQ(out({1, 1}), 0);
}

TEST(ReLUTest, NonSquareInput) {
  ReLU relu;
  Tensor input({2, 3});
  input({0, 0}) = -1; input({0, 1}) = 2; input({0, 2}) = -3;
  input({1, 0}) = 4;  input({1, 1}) = -5; input({1, 2}) = 6;
  Tensor out = relu.forward(input);
  EXPECT_FLOAT_EQ(out({0, 0}), 0);
  EXPECT_FLOAT_EQ(out({0, 1}), 2);
  EXPECT_FLOAT_EQ(out({0, 2}), 0);
  EXPECT_FLOAT_EQ(out({1, 0}), 4);
  EXPECT_FLOAT_EQ(out({1, 1}), 0);
  EXPECT_FLOAT_EQ(out({1, 2}), 6);
}

TEST(ReLUTest, OutputShapeMatchesInput) {
  ReLU relu;
  Tensor input({3, 5});
  Tensor out = relu.forward(input);
  EXPECT_EQ(out.shape(), std::vector<int>({3, 5}));
}

TEST(ReLUTest, OutputShapeMatches3D) {
  ReLU relu;
  Tensor input({2, 3, 4});
  Tensor out = relu.forward(input);
  EXPECT_EQ(out.shape(), std::vector<int>({2, 3, 4}));
  EXPECT_EQ(out.size(), 24);
}
