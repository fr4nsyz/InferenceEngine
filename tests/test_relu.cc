#include "../include/ReLU.h"
#include <gtest/gtest.h>

TEST(ReLUTest, PositiveValuesPassThrough) {
  ReLU relu;
  Matrix input(1, 3);
  input(0, 0) = 1; input(0, 1) = 2.5; input(0, 2) = 3;
  Matrix out = relu.forward(input);
  EXPECT_FLOAT_EQ(out(0, 0), 1);
  EXPECT_FLOAT_EQ(out(0, 1), 2.5);
  EXPECT_FLOAT_EQ(out(0, 2), 3);
}

TEST(ReLUTest, NegativeBecomeZero) {
  ReLU relu;
  Matrix input(1, 3);
  input(0, 0) = -1; input(0, 1) = -2.5; input(0, 2) = -3;
  Matrix out = relu.forward(input);
  EXPECT_FLOAT_EQ(out(0, 0), 0);
  EXPECT_FLOAT_EQ(out(0, 1), 0);
  EXPECT_FLOAT_EQ(out(0, 2), 0);
}

TEST(ReLUTest, ZeroStaysZero) {
  ReLU relu;
  Matrix input(1, 2);
  input(0, 0) = 0; input(0, 1) = 0;
  Matrix out = relu.forward(input);
  EXPECT_FLOAT_EQ(out(0, 0), 0);
  EXPECT_FLOAT_EQ(out(0, 1), 0);
}

TEST(ReLUTest, MixedValues) {
  ReLU relu;
  Matrix input(2, 2);
  input(0, 0) = -1; input(0, 1) = 0;
  input(1, 0) = 5;  input(1, 1) = -3;
  Matrix out = relu.forward(input);
  EXPECT_FLOAT_EQ(out(0, 0), 0);
  EXPECT_FLOAT_EQ(out(0, 1), 0);
  EXPECT_FLOAT_EQ(out(1, 0), 5);
  EXPECT_FLOAT_EQ(out(1, 1), 0);
}

TEST(ReLUTest, NonSquareInput) {
  ReLU relu;
  Matrix input(2, 3);
  input(0, 0) = -1; input(0, 1) = 2; input(0, 2) = -3;
  input(1, 0) = 4;  input(1, 1) = -5; input(1, 2) = 6;
  Matrix out = relu.forward(input);
  EXPECT_FLOAT_EQ(out(0, 0), 0);
  EXPECT_FLOAT_EQ(out(0, 1), 2);
  EXPECT_FLOAT_EQ(out(0, 2), 0);
  EXPECT_FLOAT_EQ(out(1, 0), 4);
  EXPECT_FLOAT_EQ(out(1, 1), 0);
  EXPECT_FLOAT_EQ(out(1, 2), 6);
}

TEST(ReLUTest, OutputShapeMatchesInput) {
  ReLU relu;
  Matrix input(3, 5);
  Matrix out = relu.forward(input);
  EXPECT_EQ(out._rows, 3);
  EXPECT_EQ(out._cols, 5);
}
