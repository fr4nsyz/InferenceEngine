#include "../include/Add.h"
#include <gtest/gtest.h>
#include <vector>

TEST(AddTest, TwoMatrices) {
  Add add;
  Matrix a(2, 2);
  a(0, 0) = 1; a(0, 1) = 2;
  a(1, 0) = 3; a(1, 1) = 4;

  Matrix b(2, 2);
  b(0, 0) = 5; b(0, 1) = 6;
  b(1, 0) = 7; b(1, 1) = 8;

  std::vector<Matrix> inputs = {a, b};
  Matrix out = add.forward(inputs);

  EXPECT_FLOAT_EQ(out(0, 0), 6);
  EXPECT_FLOAT_EQ(out(0, 1), 8);
  EXPECT_FLOAT_EQ(out(1, 0), 10);
  EXPECT_FLOAT_EQ(out(1, 1), 12);
}

TEST(AddTest, ThreeMatrices) {
  Add add;
  Matrix a(2, 2);
  a(0, 0) = 1; a(0, 1) = 2;
  a(1, 0) = 3; a(1, 1) = 4;

  Matrix b(2, 2);
  b(0, 0) = 5; b(0, 1) = 6;
  b(1, 0) = 7; b(1, 1) = 8;

  Matrix c(2, 2);
  c(0, 0) = 9;  c(0, 1) = 10;
  c(1, 0) = 11; c(1, 1) = 12;

  std::vector<Matrix> inputs = {a, b, c};
  Matrix out = add.forward(inputs);

  EXPECT_FLOAT_EQ(out(0, 0), 15);
  EXPECT_FLOAT_EQ(out(0, 1), 18);
  EXPECT_FLOAT_EQ(out(1, 0), 21);
  EXPECT_FLOAT_EQ(out(1, 1), 24);
}

TEST(AddTest, SingleMatrix) {
  Add add;
  Matrix a(1, 3);
  a(0, 0) = 10; a(0, 1) = 20; a(0, 2) = 30;

  std::vector<Matrix> inputs = {a};
  Matrix out = add.forward(inputs);

  EXPECT_FLOAT_EQ(out(0, 0), 10);
  EXPECT_FLOAT_EQ(out(0, 1), 20);
  EXPECT_FLOAT_EQ(out(0, 2), 30);
}

TEST(AddTest, NonSquareMatrices) {
  Add add;
  Matrix a(2, 3);
  a(0, 0) = 1; a(0, 1) = 2; a(0, 2) = 3;
  a(1, 0) = 4; a(1, 1) = 5; a(1, 2) = 6;

  Matrix b(2, 3);
  b(0, 0) = 6; b(0, 1) = 5; b(0, 2) = 4;
  b(1, 0) = 3; b(1, 1) = 2; b(1, 2) = 1;

  std::vector<Matrix> inputs = {a, b};
  Matrix out = add.forward(inputs);

  EXPECT_FLOAT_EQ(out(0, 0), 7);
  EXPECT_FLOAT_EQ(out(0, 1), 7);
  EXPECT_FLOAT_EQ(out(0, 2), 7);
  EXPECT_FLOAT_EQ(out(1, 0), 7);
  EXPECT_FLOAT_EQ(out(1, 1), 7);
  EXPECT_FLOAT_EQ(out(1, 2), 7);
}

TEST(AddTest, OutputShape) {
  Add add;
  Matrix a(3, 5);
  Matrix b(3, 5);

  std::vector<Matrix> inputs = {a, b};
  Matrix out = add.forward(inputs);

  EXPECT_EQ(out._rows, 3);
  EXPECT_EQ(out._cols, 5);
}

TEST(AddTest, EmptyInput) {
  Add add;
  std::vector<Matrix> inputs;
  EXPECT_THROW(add.forward(inputs), std::invalid_argument);
}

TEST(AddTest, ShapeMismatch) {
  Add add;
  Matrix a(2, 2);
  Matrix b(3, 3);

  std::vector<Matrix> inputs = {a, b};
  EXPECT_THROW(add.forward(inputs), std::invalid_argument);
}
