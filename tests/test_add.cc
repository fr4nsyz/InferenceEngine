#include "../include/Add.h"
#include <gtest/gtest.h>
#include <vector>

TEST(AddTest, TwoMatrices) {
  Add add;
  Tensor a({2, 2});
  a({0, 0}) = 1; a({0, 1}) = 2;
  a({1, 0}) = 3; a({1, 1}) = 4;

  Tensor b({2, 2});
  b({0, 0}) = 5; b({0, 1}) = 6;
  b({1, 0}) = 7; b({1, 1}) = 8;

  std::vector<Tensor> inputs = {a, b};
  Tensor out = add.forward(inputs);

  EXPECT_FLOAT_EQ(out({0, 0}), 6);
  EXPECT_FLOAT_EQ(out({0, 1}), 8);
  EXPECT_FLOAT_EQ(out({1, 0}), 10);
  EXPECT_FLOAT_EQ(out({1, 1}), 12);
}

TEST(AddTest, ThreeMatrices) {
  Add add;
  Tensor a({2, 2});
  a({0, 0}) = 1; a({0, 1}) = 2;
  a({1, 0}) = 3; a({1, 1}) = 4;

  Tensor b({2, 2});
  b({0, 0}) = 5; b({0, 1}) = 6;
  b({1, 0}) = 7; b({1, 1}) = 8;

  Tensor c({2, 2});
  c({0, 0}) = 9;  c({0, 1}) = 10;
  c({1, 0}) = 11; c({1, 1}) = 12;

  std::vector<Tensor> inputs = {a, b, c};
  Tensor out = add.forward(inputs);

  EXPECT_FLOAT_EQ(out({0, 0}), 15);
  EXPECT_FLOAT_EQ(out({0, 1}), 18);
  EXPECT_FLOAT_EQ(out({1, 0}), 21);
  EXPECT_FLOAT_EQ(out({1, 1}), 24);
}

TEST(AddTest, SingleMatrix) {
  Add add;
  Tensor a({1, 3});
  a({0, 0}) = 10; a({0, 1}) = 20; a({0, 2}) = 30;

  std::vector<Tensor> inputs = {a};
  Tensor out = add.forward(inputs);

  EXPECT_FLOAT_EQ(out({0, 0}), 10);
  EXPECT_FLOAT_EQ(out({0, 1}), 20);
  EXPECT_FLOAT_EQ(out({0, 2}), 30);
}

TEST(AddTest, NonSquareMatrices) {
  Add add;
  Tensor a({2, 3});
  a({0, 0}) = 1; a({0, 1}) = 2; a({0, 2}) = 3;
  a({1, 0}) = 4; a({1, 1}) = 5; a({1, 2}) = 6;

  Tensor b({2, 3});
  b({0, 0}) = 6; b({0, 1}) = 5; b({0, 2}) = 4;
  b({1, 0}) = 3; b({1, 1}) = 2; b({1, 2}) = 1;

  std::vector<Tensor> inputs = {a, b};
  Tensor out = add.forward(inputs);

  EXPECT_FLOAT_EQ(out({0, 0}), 7);
  EXPECT_FLOAT_EQ(out({0, 1}), 7);
  EXPECT_FLOAT_EQ(out({0, 2}), 7);
  EXPECT_FLOAT_EQ(out({1, 0}), 7);
  EXPECT_FLOAT_EQ(out({1, 1}), 7);
  EXPECT_FLOAT_EQ(out({1, 2}), 7);
}

TEST(AddTest, OutputShape) {
  Add add;
  Tensor a({3, 5});
  Tensor b({3, 5});

  std::vector<Tensor> inputs = {a, b};
  Tensor out = add.forward(inputs);

  EXPECT_EQ(out.shape(), std::vector<int>({3, 5}));
}

TEST(AddTest, EmptyInput) {
  Add add;
  std::vector<Tensor> inputs;
  EXPECT_THROW(add.forward(inputs), std::invalid_argument);
}

TEST(AddTest, ShapeMismatch) {
  Add add;
  Tensor a({2, 2});
  Tensor b({3, 3});

  std::vector<Tensor> inputs = {a, b};
  EXPECT_THROW(add.forward(inputs), std::invalid_argument);
}
